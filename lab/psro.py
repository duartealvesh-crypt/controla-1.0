"""PSRO (Policy-Space Response Oracles) avec CMA-ES comme oracle — palier 1.

    population ← archétypes du panel
    répéter :
      1. EGTA          jouer les profils manquants → table des gains
      2. méta-solveur  réplicateur → mélange d'équilibre σ
      3. oracle        CMA-ES : meilleure réponse à σ (adversaires tirés selon σ)
      4. test          gain de la meilleure réponse contre σ (référence : 1,5)
                       net → on l'ajoute ; sinon STOP

Deux exécutions à comparer : diplomatie libre, et diplomatie gelée
(`--frozen-diplomacy`, propose = accepte = 0 pour tous). Attention : une
diplomatie qui ne change pas l'équilibre ne prouve PAS qu'elle ne vaut rien
(chasse au cerf) — c'est le rôle des tests de déviation à deux, à venir.

    python -m lab.psro --name libre
    python -m lab.psro --name gelee --frozen-diplomacy
    python -m lab.psro --name essai --smoke
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
EGTA_SEED_BASE = 10_000


class Settings:
    def __init__(self, **values):
        self.iterations = 8
        self.egta_seeds = 3
        self.popsize = 0            # 0 → 2 × le défaut de CMA-ES
        self.generations = 20
        self.scenarios = 16         # parties par candidat et par génération
        self.eval_games = 128
        self.threshold = 0.10       # gain minimal (points de rang) pour ajouter
        self.sigma0 = 0.8
        self.seed = 0
        self.panel = list(PANEL)
        self.__dict__.update({k: v for k, v in values.items() if v is not None})


# ── étapes ──────────────────────────────────────────────────────────────────

def fill_table(runner: Runner, table: PayoffTable, population: list[dict], s: Settings) -> int:
    """Joue tous les profils encore absents de la table."""
    missing = table.missing(len(population))
    games, layout = [], []
    for profile in missing:
        for seating in seatings(profile):
            for k in range(s.egta_seeds):
                games.append(([population[i]["params"] for i in seating], EGTA_SEED_BASE + k))
                layout.append(seating)
    for seating, points in zip(layout, runner.play(games)):
        table.record(seating, points)
    return len(games)


def scenarios(rng, sigma, count: int) -> list[tuple]:
    """(seed, trio d'adversaires tirés selon σ, position du joueur évalué)."""
    k = len(sigma)
    return [
        (int(rng.integers(1, 2**31 - 1)),
         [int(i) for i in rng.choice(k, size=3, p=sigma)],
         int(rng.integers(4)))
        for _ in range(count)
    ]


def _seats(candidate: dict, trio: list[int], position: int, population: list[dict]) -> list:
    seats = [population[i]["params"] for i in trio]
    seats.insert(position, candidate)
    return seats


def score_against(runner, candidates: list[dict], cases: list[tuple], population) -> np.ndarray:
    """Points de rang moyens de chaque candidat, sur les MÊMES cas (même seeds,
    mêmes adversaires, même position) : seule la stratégie diffère."""
    games = [
        (_seats(candidate, trio, position, population), seed)
        for candidate in candidates for seed, trio, position in cases
    ]
    results = runner.play(games)
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
    default = 4 + int(3 * math.log(space.dim))
    options["popsize"] = s.popsize or 2 * default
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


def evaluate(runner, candidate: dict, population, sigma, s: Settings, rng) -> tuple[float, float]:
    cases = scenarios(rng, sigma, s.eval_games)
    points = score_against(runner, [candidate], cases, population)[0]
    return float(points.mean() - NEUTRAL), float(points.std(ddof=1) / math.sqrt(len(points)))


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
            played = fill_table(runner, table, population, s)
            sigma = table.replicator(len(population))
            fitness = table.fitness(sigma)
            regret = table.regret(sigma)
            log(f"\n── itération {it} · {len(population)} stratégies · "
                f"EGTA {played} parties ──")
            log(_sigma_line(population, sigma, fitness))
            log(f"  regret dans la population : {regret:.3f}")

            start = population[int(np.argmax(fitness))]["params"]
            label = f"br{it}"
            log(f"  oracle : CMA-ES depuis {population[int(np.argmax(fitness))]['label']}")
            br = best_response(runner, space, population, sigma, start, s, rng, label, log)
            gain, se = evaluate(runner, br, population, sigma, s, rng)
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
            save("running", sigma)
        else:
            fill_table(runner, table, population, s)
            sigma = table.replicator(len(population))
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

SMOKE = dict(iterations=1, egta_seeds=1, popsize=6, generations=2, scenarios=4,
             eval_games=8, panel=["cooperator", "conqueror", "turtle"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lab.psro", description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--frozen-diplomacy", action="store_true")
    parser.add_argument("--target-rule", default="none")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--egta-seeds", type=int)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--eval-games", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--smoke", action="store_true", help="réglages minuscules, pour tester")
    args = parser.parse_args(argv)

    values = {key: getattr(args, key) for key in (
        "iterations", "egta_seeds", "popsize", "generations", "scenarios", "eval_games",
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
