"""Plusieurs équilibres ? Départs multiples, α-Rank, duel entre exécutions.

    python -m lab.equilibria runs/lab/psro-libre-s0              # table seule, sans partie
    python -m lab.equilibria --duel runs/lab/psro-libre-s0 runs/lab/psro-gelee-s0

**Départs multiples.** Le réplicateur part d'un point ; il aboutit à l'équilibre
dont il est dans le bassin. On le relance depuis de nombreux points tirés au
hasard : plusieurs points d'arrivée = plusieurs équilibres, et la part des
départs qui y mènent dit la taille de chaque bassin.

**α-Rank** (Omidshafiei et al. 2019). Une population de m = 4 joueurs, tous de la
même stratégie i ; un mutant j apparaît et se répand avec une probabilité qui
croît avec son avantage (règle de Fermi, intensité α). Avec m = 4, une
population EST un profil de la table : il ne faut que les cases à 2 stratégies.
La distribution stationnaire de cette chaîne classe les stratégies, existe
toujours, est unique, et gère les cycles. α grand → seuls les gains comptent.

**Duel.** Deux exécutions (graines ou variantes différentes) ont-elles trouvé des
équilibres équivalents ? 2 sièges tirés selon σ_A, 2 selon σ_B, sur des cartes
neuves. Le jeu étant à somme constante, un camp à +x prend x à l'autre.

**Note.** Les points de rang font 6 par partie : tout équilibre SYMÉTRIQUE donne
1,5 à chacun. « Un équilibre meilleur pour tous » n'a donc pas de sens ici ; la
chasse au cerf prend la forme d'une coalition qui gagne aux dépens des autres
(voir `lab.coalitions`).
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from lab.metagame import SEATS, PayoffTable
from lab.report import NEUTRAL, final_sigma, load_state
from lab.runner import Runner


# ── départs multiples ───────────────────────────────────────────────────────

def multistart(table: PayoffTable, k: int, starts: int = 64, rng=None,
               iterations: int = 5000, merge: float = 0.1) -> list[dict]:
    """Points d'arrivée du réplicateur depuis `starts` départs (Dirichlet(1)),
    regroupés quand ils sont à moins de `merge` en distance L1."""
    rng = rng or np.random.default_rng(0)
    found: list[dict] = []
    for _ in range(starts):
        x0 = rng.dirichlet(np.ones(k))
        sigma = table.replicator(k, x0=x0, iterations=iterations)
        for item in found:
            if np.abs(item["sigma"] - sigma).sum() < merge:
                item["hits"] += 1
                break
        else:
            found.append({"sigma": sigma, "hits": 1, "regret": table.regret(sigma)})
    for item in found:
        item["share"] = item["hits"] / starts
    return sorted(found, key=lambda item: -item["hits"])


# ── α-Rank ──────────────────────────────────────────────────────────────────

def fixation(table: PayoffTable, resident: int, mutant: int, alpha: float) -> float:
    """Probabilité qu'un mutant j envahisse une population de 4 résidents i."""
    m = SEATS
    # Δ_k = gain du mutant − gain du résident quand il y a k mutants.
    deltas = []
    for k in range(1, m):
        profile = (mutant,) * k + (resident,) * (m - k)
        deltas.append(table.payoff(profile, mutant) - table.payoff(profile, resident))
    # ρ = 1 / (1 + Σ_l Π_{k ≤ l} e^{−α Δ_k}), calculé en log pour éviter l'overflow.
    logs = np.concatenate([[0.0], np.cumsum(-alpha * np.asarray(deltas))])
    top = logs.max()
    return float(math.exp(-(top + math.log(np.exp(logs - top).sum()))))


def alpha_rank(table: PayoffTable, k: int, alpha: float) -> np.ndarray:
    """Distribution stationnaire de la chaîne des populations monomorphes."""
    if k == 1:
        return np.ones(1)
    chain = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                chain[i, j] = fixation(table, i, j, alpha) / (k - 1)
        chain[i, i] = 1.0 - chain[i].sum()
    # π C = π, Σ π = 1 : système linéaire surdéterminé, résolu aux moindres carrés.
    system = np.vstack([chain.T - np.eye(k), np.ones(k)])
    target = np.concatenate([np.zeros(k), [1.0]])
    pi = np.linalg.lstsq(system, target, rcond=None)[0]
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


# ── duel entre exécutions ───────────────────────────────────────────────────

def duel(runner: Runner, a: dict, b: dict, games: int, rng) -> tuple[float, float]:
    """Points moyens du camp A (2 sièges selon σ_A) contre B (2 selon σ_B), − 1,5."""
    sides = []
    for state in (a, b):
        sigma = final_sigma(state)
        sides.append(([p["params"] for p in state["population"][:len(sigma)]], sigma))
    batch, camps = [], []
    for _ in range(games):
        seats, camp = [], []
        for side, (params, sigma) in enumerate(sides):
            for i in rng.choice(len(sigma), size=2, p=sigma):
                seats.append(params[int(i)])
                camp.append(side)
        order = rng.permutation(SEATS)
        batch.append(([seats[o] for o in order], int(rng.integers(1, 2**31 - 1))))
        camps.append([camp[o] for o in order])
    results = runner.play(batch)
    scores = np.array([
        np.mean([p for p, c in zip(points, camp) if c == 0])
        for points, camp in zip(results, camps)
    ])
    return float(scores.mean() - NEUTRAL), float(scores.std(ddof=1) / math.sqrt(len(scores)))


# ── ligne de commande ───────────────────────────────────────────────────────

def render(state: dict, starts: int, alphas: list[float]) -> str:
    table, k = state["payoffs"], state["complete"]
    labels = [p["label"] for p in state["population"][:k]]

    def mix(x) -> str:
        return ", ".join(f"{labels[i]} {x[i]:.0%}" for i in np.argsort(-x) if x[i] > 0.005)

    lines = [f"═══ {state['name']} · {k} stratégies", "", f"  réplicateur, {starts} départs :"]
    for item in multistart(table, k, starts):
        lines.append(f"    {item['share']:>4.0%} des départs → {mix(item['sigma'])}"
                     f"  (regret {item['regret']:.3f})")
    lines += ["", "  α-Rank :"]
    for alpha in alphas:
        lines.append(f"    α = {alpha:<5g} {mix(alpha_rank(table, k, alpha))}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lab.equilibria",
                                     description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--starts", type=int, default=64)
    parser.add_argument("--alphas", default="1,5,20,100")
    parser.add_argument("--duel", action="store_true",
                        help="duel entre les deux premières exécutions (joue des parties)")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    states = [load_state(path) for path in args.runs]
    if args.duel:
        if len(states) != 2:
            parser.error("--duel demande exactement deux exécutions")
        with Runner(args.workers) as runner:
            gain, se = duel(runner, states[0], states[1], args.games, np.random.default_rng(args.seed))
        a, b = states[0]["name"], states[1]["name"]
        print(f"{a} contre {b} : {gain:+.3f} ± {se:.3f} point de rang par siège pour {a}")
        return 0
    alphas = [float(a) for a in args.alphas.split(",")]
    for state in states:
        print(render(state, args.starts, alphas))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
