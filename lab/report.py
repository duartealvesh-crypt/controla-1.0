"""Résumé d'exécutions PSRO (`state.json`), et contrôle de cohérence de la table.

    python -m lab.report runs/lab/psro-libre-s0 runs/lab/psro-gelee-s0 …

**Cohérence.** Une meilleure réponse est ajoutée parce qu'elle bat σ sur 128
parties fraîches. La table, une fois complétée, doit dire la même chose : son
gain contre le σ de son itération, lu dans la table finale, doit être proche du
gain mesuré. Un grand écart = table encore trop bruitée (le défaut de la
première version, où +0,26 mesuré devenait −0,05 dans la table).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lab.metagame import PayoffTable
from lab.strategy import describe

NEUTRAL = 1.5


def load_state(path) -> dict:
    """Charge un `state.json` (chemin du fichier ou de son dossier) et sa table."""
    path = Path(path)
    if path.is_dir():
        path = path / "state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["path"] = str(path)
    state["payoffs"] = PayoffTable.from_json(state["table"])
    # Une exécution coupée en route peut avoir ajouté une stratégie dont les
    # profils ne sont pas encore joués : on n'analyse que la partie complète.
    k = len(state["population"])
    while k > 1 and state["payoffs"].missing(k):
        k -= 1
    state["complete"] = k
    return state


def padded(weights: dict[str, float], population: list[dict], k: int) -> np.ndarray:
    """Un mélange {label: poids} remis dans l'ordre des k premières stratégies."""
    x = np.array([weights.get(p["label"], 0.0) for p in population[:k]], dtype=float)
    return x / x.sum()


def final_sigma(state: dict) -> np.ndarray:
    k = state["complete"]
    return state["payoffs"].replicator(k)


def consistency(state: dict) -> list[dict]:
    table, k, population = state["payoffs"], state["complete"], state["population"]
    labels = [p["label"] for p in population[:k]]
    rows = []
    for item in state["iterations"]:
        sigma = padded(item["sigma"], population, k)
        label = f"br{item['iteration']}"
        if item["accepted"] and label in labels:
            rows.append({
                "label": label, "measured": item["br_gain"] + NEUTRAL, "se": item["br_se"],
                "table": float(table.fitness(sigma)[labels.index(label)]),
            })
        label = f"eq{item['iteration']}"
        if item.get("team_accepted") and label in labels:
            rows.append({
                "label": f"{label} (équipe)", "measured": item["team_mean"], "se": item["team_se"],
                "table": float(table.team_fitness(sigma)[labels.index(label)]),
            })
    return rows


def render(state: dict) -> str:
    population, k = state["population"], state["complete"]
    lines = [
        f"═══ {state['name']} · {'diplomatie gelée' if state['frozen_diplomacy'] else 'diplomatie libre'}"
        f" · {state['status']} · {state['games']} parties · {state['seconds'] / 3600:.1f} h",
        "",
        "  it.  K   σ (support)                                   regret  meilleures réponses",
    ]
    for item in state["iterations"]:
        support = ", ".join(f"{name} {w:.0%}" for name, w in
                            sorted(item["sigma"].items(), key=lambda kv: -kv[1]) if w > 0.005)
        verdict = "ajoutée" if item["accepted"] else "rejetée"
        line = (f"  {item['iteration']:>3} {item['strategies']:>3}   {support[:44]:<44}  "
                f"{item['regret']:.3f}   solo {item['br_gain']:+.3f} ± {item['br_se']:.3f} {verdict}")
        if "team_gain" in item:
            verdict = "ajoutée" if item["team_accepted"] else "rejetée"
            line += f" · équipe {item['team_gain']:+.3f} ± {item['team_se']:.3f} {verdict}"
        lines.append(line)

    sigma = final_sigma(state)
    table = state["payoffs"]
    lines += ["", f"  σ final (table complète sur {k} stratégies, regret {table.regret(sigma):.3f}) :"]
    for i in np.argsort(-sigma):
        if sigma[i] <= 0.005:
            break
        lines.append(f"    {sigma[i]:>4.0%}  {population[i]['label']:<18} "
                     f"{describe(population[i]['params'])}")

    rows = consistency(state)
    if rows:
        lines += ["", "  cohérence (points par siège contre le σ de son itération) :  mesuré   table"]
        for row in rows:
            flag = "  ← écart" if abs(row["measured"] - row["table"]) > 2.5 * row["se"] + 0.1 else ""
            lines.append(f"    {row['label']:<56} {row['measured']:.3f}    {row['table']:.3f}{flag}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lab.report", description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", help="dossiers psro-* ou fichiers state.json")
    args = parser.parse_args(argv)
    for path in args.runs:
        print(render(load_state(path)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
