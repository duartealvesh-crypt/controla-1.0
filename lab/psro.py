"""PSRO (Policy-Space Response Oracles) avec CMA-ES comme oracle — palier 1.

    population ← archétypes du panel
    répéter :
      1. EGTA          jouer les profils manquants → table des gains
      2. méta-solveur  réplicateur → mélange d'équilibre σ ; préciser les cases
                       qui pèsent sous σ, recalculer σ (jusqu'à stabilité)
      3. oracle        CMA-ES : meilleure réponse à σ (adversaires tirés selon σ)
      4. test          gain de la meilleure réponse contre σ (référence : 1,5)
                       net → on l'ajoute (et ses parties de test vont dans la
                       table) ; sinon STOP

**Le bruit de la table.** Chaque case est une moyenne de parties. La première
version jouait tous les profils sur les 3 mêmes cartes : ±0,3 point par case, et
une erreur commune à toutes les stratégies. Une meilleure réponse mesurée à
+0,26 contre σ sur 128 parties fraîches y paraissait perdante, et σ n'a plus
bougé de 6 itérations. Désormais : une base légère (une partie par rotation,
graines tirées au hasard) pour tous les profils, puis `refine_target` sièges dans
chaque case dont le trio d'adversaires a une probabilité ≥ `refine_pmin` sous σ.

Deux exécutions à comparer : diplomatie libre, et diplomatie gelée
(`--frozen-diplomacy`, propose = accepte = 0 pour tous). Attention : une
diplomatie qui ne change pas l'équilibre ne prouve PAS qu'elle ne vaut rien
(chasse au cerf) — c'est le rôle des tests de déviation à deux, à venir.

    python -m lab.psro --name libre
    python -m lab.psro --name gelee --frozen-diplomacy
    python -m lab.psro --name essai --smoke
    python -m lab.psro --name libre --workers 6     # garder des cœurs libres
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from controla import api
from lab.metagame import PayoffTable, seatings
from lab.runner import Runner
from lab.strategy import CLIP, Space, describe

PANEL = ("cooperator", "betrayer", "conqueror", "tit_for_tat", "expansionist",
         "coalition_builder", "turtle")
NEUTRAL = 1.5   # points de rang moyens d'un joueur contre des copies du mélange


class Settings:
    def __init__(self, **values):
        self.iterations = 8
        self.base_games = 4         # parties par profil à sa création (1 par rotation)
        self.refine_target = 32     # sièges visés dans chaque case qui pèse sous σ
        self.refine_pmin = 0.03     # probabilité minimale du trio pour peser
        self.refine_rounds = 3      # allers-retours σ → précision → σ
        self.popsize = 0            # 0 → défaut de CMA-ES (4 + 3 ln d)
        self.generations = 12
        self.scenarios = 16         # parties par candidat et par génération
        self.eval_games = 128
        self.threshold = 0.10       # gain minimal (points de rang) pour ajouter
        self.sigma0 = 0.8
        self.seed = 0
        self.panel = list(PANEL)
        self.__dict__.update({k: v for k, v in values.items() if v is not None})


# ── étapes ──────────────────────────────────────────────────────────────────

def _seed(rng) -> int:
    return int(rng.integers(1, 2**31 - 1))


def _play_into(runner, table: PayoffTable, population, layout: list[list[int]], rng) -> int:
    """Joue chaque placement sur une carte neuve et verse les points dans la table."""
    games = [([population[i]["params"] for i in seating], _seed(rng)) for seating in layout]
    for seating, points in zip(layout, runner.play(games)):
        table.record(seating, points)
    return len(games)


def fill_table(runner: Runner, table: PayoffTable, population: list[dict], s: Settings,
               rng) -> int:
    """Base légère pour les profils encore absents : `base_games` parties, en
    tournant les sièges, chacune sur sa propre carte."""
    layout = [
        seatings(profile)[g % 4]
        for profile in table.missing(len(population))
        for g in range(s.base_games)
    ]
    return _play_into(runner, table, population, layout, rng)


def refine(runner, table: PayoffTable, population, sigma, s: Settings, rng) -> int:
    """Complète les cases qui pèsent sous σ jusqu'à `refine_target` sièges."""
    needed: dict[tuple, int] = {}
    for profile, i in table.relevant(sigma, s.refine_pmin):
        missing = s.refine_target - table.count(profile, i)
        if missing > 0:
            per_game = profile.count(i)       # sièges de i gagnés par partie
            needed[profile] = max(needed.get(profile, 0), -(-missing // per_game))
    layout = [
        seatings(profile)[g % 4]
        for profile, games in needed.items()
        for g in range(games)
    ]
    return _play_into(runner, table, population, layout, rng)


def solve(runner, table: PayoffTable, population, s: Settings, rng):
    """σ, puis précision des cases qui comptent sous σ, puis σ recalculé…"""
    played = 0
    sigma = table.replicator(len(population))
    for _ in range(s.refine_rounds):
        added = refine(runner, table, population, sigma, s, rng)
        played += added
        if added == 0:
            break
        sigma = table.replicator(len(population))
    return sigma, played


def scenarios(rng, sigma, count: int) -> list[tuple]:
    """(seed, trio d'adversaires tirés selon σ, position du joueur évalué)."""
    k = len(sigma)
    return [
        (_seed(rng), [int(i) for i in rng.choice(k, size=3, p=sigma)], int(rng.integers(4)))
        for _ in range(count)
    ]


def _seats(candidate: dict, trio: list[int], position: int, population: list[dict]) -> list:
    seats = [population[i]["params"] for i in trio]
    seats.insert(position, candidate)
    return seats


def play_cases(runner, candidates: list[dict], cases: list[tuple], population) -> list:
    """Chaque candidat sur les MÊMES cas (même seeds, mêmes adversaires, même
    position) : seule la stratégie diffère. Rend les points des 4 sièges."""
    games = [
        (_seats(candidate, trio, position, population), seed)
        for candidate in candidates for seed, trio, position in cases
    ]
    return runner.play(games)


def score_against(runner, candidates: list[dict], cases: list[tuple], population) -> np.ndarray:
    """Points de rang de chaque candidat : tableau (candidats × cas)."""
    results = play_cases(runner, candidates, cases, population)
    points = np.array([
        result[position]
        for result, (_, _, position) in zip(results, cases * len(candidates))
    ], dtype=float)
    return points.reshape(len(candidates), len(cases))


def best_response(runner, space: Space, population, sigma, start: dict, s: Settings,
                  rng, label: str, log) -> dict:
    import cma

    options = {
        "bounds": [-CLIP, CLIP],
        "maxiter": s.generations,
        "seed": int(rng.integers(1, 2**31 - 1)),
        "verbose": -9,
    }
    options["popsize"] = s.popsize or 4 + int(3 * math.log(space.dim))
    es = cma.CMAEvolutionStrategy(space.encode(start), s.sigma0, options)
    generation = 0
    while not es.stop():
        generation += 1
        cases = scenarios(rng, sigma, s.scenarios)
        xs = es.ask()
        candidates = [space.decode(x, label) for x in xs]
        means = score_against(runner, candidates, cases, population).mean(axis=1)
        es.tell(xs, [-m for m in means])
        log(f"    génération {generation:>2} · meilleur {means.max():.2f} · "
            f"moyenne {means.mean():.2f} · σ {es.sigma:.2f}")
    return space.decode(es.result.xfavorite, label)


def evaluate(runner, candidate: dict, population, sigma, s: Settings, rng):
    """Gain contre σ sur des parties fraîches : (gain − 1,5, erreur type, parties).

    Les parties sont rendues pour être versées dans la table si le candidat
    entre dans la population : ce sont justement les cases qui pèsent sous σ."""
    cases = scenarios(rng, sigma, s.eval_games)
    results = play_cases(runner, [candidate], cases, population)
    points = np.array([r[pos] for r, (_, _, pos) in zip(results, cases)], dtype=float)
    gain = float(points.mean() - NEUTRAL)
    se = float(points.std(ddof=1) / math.sqrt(len(points)))
    return gain, se, list(zip(cases, results))


def record_cases(table: PayoffTable, index: int, games) -> None:
    """Verse les parties d'évaluation d'une stratégie devenue `index`."""
    for (_, trio, position), points in games:
        seating = list(trio)
        seating.insert(position, index)
        table.record(seating, points)


# ── boucle ──────────────────────────────────────────────────────────────────

def run(name: str, space: Space, s: Settings, workers=None, log=print) -> dict:
    out_dir = Path("runs/lab") / f"psro-{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(s.seed)

    population = [
        {"label": label, "params": space.adapt(api.archetype_params(label)), "origin": "panel"}
        for label in s.panel
    ]
    table = PayoffTable()
    iterations: list[dict] = []
    started = time.perf_counter()

    def save(status: str, sigma=None):
        state = {
            "name": name, "status": status, "frozen_diplomacy": space.frozen,
            "target_rule": space.target_rule, "settings": vars(s),
            "population": population, "iterations": iterations,
            "sigma": None if sigma is None else [float(v) for v in sigma],
            "table": table.to_json(), "games": runner.games,
            "seconds": round(time.perf_counter() - started, 1),
        }
        (out_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1),
                                            encoding="utf-8")

    with Runner(workers) as runner:
        for it in range(1, s.iterations + 1):
            t0 = time.perf_counter()
            played = fill_table(runner, table, population, s, rng)
            sigma, refined = solve(runner, table, population, s, rng)
            fitness = table.fitness(sigma)
            regret = table.regret(sigma)
            log(f"\n── itération {it} · {len(population)} stratégies · "
                f"EGTA {played} + {refined} parties de précision ──")
            log(_sigma_line(population, sigma, fitness))
            log(f"  regret dans la population : {regret:.3f}")

            start = population[int(np.argmax(fitness))]["params"]
            label = f"br{it}"
            log(f"  oracle : CMA-ES depuis {population[int(np.argmax(fitness))]['label']}")
            br = best_response(runner, space, population, sigma, start, s, rng, label, log)
            gain, se, games = evaluate(runner, br, population, sigma, s, rng)
            accepted = gain > max(s.threshold, 2 * se)
            log(f"  meilleure réponse : gain {gain:+.3f} ± {se:.3f} contre σ "
                f"→ {'AJOUTÉE' if accepted else 'rejetée : STOP'}")
            log(f"  {describe(br)}")

            iterations.append({
                "iteration": it, "strategies": len(population),
                "sigma": {p["label"]: float(v) for p, v in zip(population, sigma)},
                "fitness": {p["label"]: float(v) for p, v in zip(population, fitness)},
                "regret": regret, "br_gain": gain, "br_se": se, "accepted": accepted,
                "br": br, "seconds": round(time.perf_counter() - t0, 1),
            })
            if not accepted:
                save("converged", sigma)
                break
            population.append({"label": label, "params": br, "origin": f"oracle {it}"})
            record_cases(table, len(population) - 1, games)
            save("running", sigma)
        else:
            fill_table(runner, table, population, s, rng)
            sigma, _ = solve(runner, table, population, s, rng)
            log("\n── fin (nombre d'itérations atteint) ──")
            log(_sigma_line(population, sigma, table.fitness(sigma)))
            save("max_iterations", sigma)

        log(f"\n{runner.games} parties en {time.perf_counter() - started:.0f} s "
            f"→ {out_dir / 'state.json'}")
    return json.loads((out_dir / "state.json").read_text(encoding="utf-8"))


def _sigma_line(population, sigma, fitness) -> str:
    order = np.argsort(-np.asarray(sigma))
    support = [f"{population[i]['label']} {sigma[i]:.0%}" for i in order if sigma[i] > 0.005]
    best = [f"{population[i]['label']} {fitness[i]:.2f}" for i in np.argsort(-fitness)[:4]]
    return f"  σ : {', '.join(support)}\n  gain contre σ : {', '.join(best)}"


# ── ligne de commande ───────────────────────────────────────────────────────

SMOKE = dict(iterations=2, base_games=4, refine_target=6, popsize=6, generations=2,
             scenarios=4, eval_games=8,
             panel=["cooperator", "conqueror", "turtle"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lab.psro", description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--frozen-diplomacy", action="store_true")
    parser.add_argument("--target-rule", default="none")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--base-games", type=int)
    parser.add_argument("--refine-target", type=int)
    parser.add_argument("--refine-pmin", type=float)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--eval-games", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int,
                        help="processus de calcul (défaut : cœurs − 1, en priorité basse)")
    parser.add_argument("--smoke", action="store_true", help="réglages minuscules, pour tester")
    args = parser.parse_args(argv)

    values = {key: getattr(args, key) for key in (
        "iterations", "base_games", "refine_target", "refine_pmin", "popsize", "generations", "scenarios", "eval_games",
        "threshold", "seed")}
    if args.smoke:
        values = {**SMOKE, **{k: v for k, v in values.items() if v is not None}}
    settings = Settings(**values)
    space = Space(frozen_diplomacy=args.frozen_diplomacy, target_rule=args.target_rule)
    run(args.name, space, settings, workers=args.workers,
        log=lambda line: print(line, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
