"""Tests de déviation à deux : une coalition gagne-t-elle là où un joueur seul ne
gagne rien ?

    python -m lab.coalitions runs/lab/psro-libre-s0            # les 3 pactes
    python -m lab.coalitions runs/lab/psro-libre-s0 --pact econ_alliance
    python -m lab.coalitions runs/lab/psro-libre-s0 --pact war_alliance --games 600
    python -m lab.coalitions runs/lab/psro-gelee-s0 --pact econ_alliance --base br3

PSRO ne teste que des déviations UNILATÉRALES. Or un engagement exige deux
promesses miroir : la diplomatie ne peut payer que dans une déviation conjointe.

**Protocole** (nombres aléatoires communs). Chaque partie tire 4 stratégies selon
σ et deux sièges au hasard. Elle est jouée deux fois, même carte, mêmes
adversaires :
  - référence : la partie telle quelle ;
  - pacte : les deux sièges choisis gardent leur stratégie (ou `--base`) mais ont
    convenu d'avance de sceller `--pact` entre eux et de ne pas se frapper.
    `war_alliance` vise le leader visible hors pacte.
  - `non_aggression` ne scelle rien : seulement ne pas se frapper. C'est la
    référence — gain(alliance) − gain(non_aggression) = ce que l'engagement
    lui-même rapporte, au-delà d'une simple trêve.
Gain de la coalition = points moyens de la paire avec pacte − sans pacte.

**Lecture.** Gain net > 0 : l'équilibre n'est PAS robuste aux coalitions — la
diplomatie a une valeur que PSRO ne voit pas (chasse au cerf). Gain nul ou
négatif : l'équilibre résiste, au moins à cette coalition-là. Le jeu étant à
somme constante, le gain de la paire est pris aux deux autres.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from controla import api
from lab.report import NEUTRAL, final_sigma, load_state
from lab.runner import RANK_POINTS, Runner

PACTS = ("non_aggression", "econ_alliance", "war_alliance")


def _play(job: tuple) -> dict:
    """Une partie : points de chaque siège et engagements formés."""
    seats, seed = job
    result = api.play_game(seats, seed)
    points = [0] * len(seats)
    for item in result["ranking"]:
        points[item["faction_id"]] = RANK_POINTS[item["rank"]]
    formed = [
        (event["commitment"], sorted(event["parties"]))
        for turn in result["history"] for event in turn["events"]
        if event["kind"] == "commitment_formed"
    ]
    return {"points": points, "formed": formed}


def with_pact(params: dict, partner: int, pact: str) -> dict:
    return dict(params, partners=[partner], pact=pact, honour_rate=1.0)


def deviation_test(runner: Runner, state: dict, pact: str, games: int, rng,
                   base: str | None = None) -> dict:
    sigma = final_sigma(state)
    population = state["population"][:len(sigma)]
    labels = [p["label"] for p in population]
    if base is not None and base not in labels:
        raise ValueError(f"stratégie inconnue : {base} (connues : {', '.join(labels)})")

    cases = []
    for _ in range(games):
        drawn = [population[int(i)]["params"] for i in rng.choice(len(sigma), size=4, p=sigma)]
        a, b = (int(s) for s in rng.choice(4, size=2, replace=False))
        if base is not None:
            chosen = population[labels.index(base)]["params"]
            drawn[a], drawn[b] = chosen, chosen
        paired = list(drawn)
        paired[a] = with_pact(drawn[a], b, pact)
        paired[b] = with_pact(drawn[b], a, pact)
        cases.append((drawn, paired, a, b, int(rng.integers(1, 2**31 - 1))))

    jobs = [(drawn, seed) for drawn, _, _, _, seed in cases]
    jobs += [(paired, seed) for _, paired, _, _, seed in cases]
    results = runner.map(_play, jobs)
    reference, pacted = results[:games], results[games:]

    pair_ref, pair_pact, sealed = [], [], 0
    for (_, _, a, b, _), ref, dev in zip(cases, reference, pacted):
        pair_ref.append((ref["points"][a] + ref["points"][b]) / 2)
        pair_pact.append((dev["points"][a] + dev["points"][b]) / 2)
        sealed += any(kind == pact and parties == sorted((a, b)) for kind, parties in dev["formed"])
    pair_ref, pair_pact = np.array(pair_ref), np.array(pair_pact)
    diff = pair_pact - pair_ref
    return {
        "pact": pact, "base": base, "games": games,
        "reference": float(pair_ref.mean()), "with_pact": float(pair_pact.mean()),
        "gain": float(diff.mean()), "se": float(diff.std(ddof=1) / math.sqrt(games)),
        "sealed": sealed / games,
    }


def render(state: dict, result: dict) -> str:
    gain, se = result["gain"], result["se"]
    if gain > 2 * se:
        verdict = "la paire GAGNE : l'équilibre n'est pas robuste à cette coalition"
    elif gain < -2 * se:
        verdict = "la paire PERD : ce pacte lui coûte"
    else:
        verdict = "pas d'effet net mesurable"
    base = f" · paire = {result['base']}" if result["base"] else ""
    return "\n".join([
        f"═══ {state['name']} · pacte {result['pact']}{base} · {result['games']} parties ×2",
        f"  paire sans pacte : {result['reference']:.3f}   (référence ≈ {NEUTRAL})",
        f"  paire avec pacte : {result['with_pact']:.3f}",
        f"  gain             : {gain:+.3f} ± {se:.3f} point de rang par siège → {verdict}",
        *([] if result["pact"] == "non_aggression"
          else [f"  pacte scellé dans {result['sealed']:.0%} des parties"]),
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lab.coalitions",
                                     description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--pact", choices=(*PACTS, "all"), default="all")
    parser.add_argument("--base", help="stratégie de la paire (défaut : celles tirées selon σ)")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    pacts = PACTS if args.pact == "all" else (args.pact,)
    rng = np.random.default_rng(args.seed)
    with Runner(args.workers) as runner:
        for path in args.runs:
            state = load_state(path)
            for pact in pacts:
                print(render(state, deviation_test(runner, state, pact, args.games, rng, args.base)))
                print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
