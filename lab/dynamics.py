"""Rapport de dynamique : où en est le jeu par rapport à l'intention de design ?

Joue un panel scripté sur plusieurs seeds, dans toutes les rotations de sièges
(chaque stratégie joue chaque coin), et mesure :

* le tour du **premier contact** entre deux territoires (cible : ~5) ;
* le nombre de **paires en guerre** en fin de partie (cible : 1 à 5 sur 6) ;
* la part de parties gagnées par le **leader de mi-partie** (cible : 40–50 %) ;
* par stratégie : points de rang (3/2/1/0), victoires, actions jouées.

Aucun appel LLM. Usage :

    python -m lab.dynamics
    python -m lab.dynamics --seeds 1-24 --panel cooperator,conqueror,expansionist,turtle
    python -m lab.dynamics --set world.grid_size=10
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from controla import api

DEFAULT_PANEL = ("cooperator", "conqueror", "expansionist", "turtle")
RANK_POINTS = (3, 2, 1, 0)

# L'intention de design, chiffrée (voir notes/rl-equilibrage.md, section 7).
TARGETS = {
    "first_contact": (4.0, 6.0),        # tour médian du premier contact
    "war_pairs": (1, 5),                # paires en guerre en fin de partie
    "leader_mid_wins": (0.40, 0.50),    # part de victoires du leader de mi-partie
}


# ── une partie ──────────────────────────────────────────────────────────────

def rotations(panel: list[str]) -> list[list[str]]:
    """Les 4 rotations du panel : chaque stratégie occupe chaque coin une fois."""
    return [panel[k:] + panel[:k] for k in range(len(panel))]


def _contact(owner: list[int], size: int) -> bool:
    """Deux territoires de factions différentes se touchent-ils (4-voisinage) ?"""
    for row in range(size):
        for col in range(size):
            mine = owner[row * size + col]
            if mine < 0:
                continue
            for d_row, d_col in ((0, 1), (1, 0)):
                r, c = row + d_row, col + d_col
                if r < size and c < size:
                    other = owner[r * size + c]
                    if other >= 0 and other != mine:
                        return True
    return False


def _war_pairs(factions: list[dict]) -> set[tuple[int, int]]:
    return {
        (min(f["id"], enemy), max(f["id"], enemy))
        for f in factions for enemy in f["at_war_with"]
    }


def spearman(first: list[float], second: list[float]) -> float:
    """Corrélation de rang de Spearman (sans ex aequo)."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0] * len(values)
        for rank, index in enumerate(order):
            out[index] = rank
        return out

    a, b = ranks(first), ranks(second)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(a, b)) / (n * (n * n - 1))


def game_metrics(result: dict, mid_turn: int) -> dict:
    """Les mesures d'une partie, à partir du résultat de `api.play_game`."""
    size = int(result["world"]["size"])
    seats = result["seats"]
    history = result["history"]

    initial = result["world"].get("initial_board") or {}
    first_contact = 0 if initial.get("owner") and _contact(initial["owner"], size) else None
    first_war = None
    for turn in history:
        board = turn["board"]
        if first_contact is None and _contact(board["owner"], size):
            first_contact = turn["turn"]
        if first_war is None and _war_pairs(board["factions"]):
            first_war = turn["turn"]

    final_factions = history[-1]["board"]["factions"]
    mid = next((t for t in history if t["turn"] == mid_turn), history[len(history) // 2])
    mid_scores = {f["id"]: f["score"] for f in mid["board"]["factions"]}
    leader_mid = max(mid_scores, key=mid_scores.get)

    final_rank = {item["faction_id"]: item["rank"] for item in result["ranking"]}
    winner = next(fid for fid, rank in final_rank.items() if rank == 0)
    ids = sorted(final_rank)

    actions: dict[int, Counter] = defaultdict(Counter)
    attacks: Counter = Counter()
    for turn in history:
        for event in turn["events"]:
            if event["kind"] == "divine_action":
                actions[event["faction"]][event["action"]] += 1
            elif event["kind"] == "war_launched":
                attacks[event["faction"]] += 1

    battles = [b for turn in history for b in turn["battles"]]
    return {
        "seed": result["seed"],
        "seats": seats,
        "first_contact": first_contact,
        "first_war": first_war,
        "war_pairs": len(_war_pairs(final_factions)),
        "leader_mid": leader_mid,
        "leader_mid_won": leader_mid == winner,
        # Score à mi-partie contre rang final (rang 0 = premier → on inverse).
        "mid_final_spearman": spearman(
            [mid_scores[i] for i in ids], [-final_rank[i] for i in ids]
        ),
        "battles": len(battles),
        "eliminations": sum(len(turn["eliminations"]) for turn in history),
        "per_seat": [
            {
                "faction": fid,
                "name": seats[fid],
                "rank": final_rank[fid],
                "points": RANK_POINTS[final_rank[fid]],
                "actions": dict(actions[fid]),
                "attacks": attacks[fid],
            }
            for fid in ids
        ],
    }


def _play_one(job: tuple) -> dict:
    seats, seed, overrides, mid_turn = job
    return game_metrics(api.play_game(seats, seed, overrides), mid_turn)


# ── une campagne ────────────────────────────────────────────────────────────

def run(panel, seeds, overrides=None, mid_turn=7, workers=None, on_game=None) -> list[dict]:
    """Joue `seeds × rotations` parties, en parallèle, et rend leurs mesures."""
    jobs = [
        (seats, seed, overrides, mid_turn)
        for seed in seeds for seats in rotations(list(panel))
    ]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        games = []
        for job in jobs:
            games.append(_play_one(job))
            if on_game:
                on_game(len(games), len(jobs))
        return games
    games = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for game in pool.map(_play_one, jobs):
            games.append(game)
            if on_game:
                on_game(len(games), len(jobs))
    return games


def _share(flags) -> float:
    flags = list(flags)
    return sum(flags) / len(flags) if flags else 0.0


def summarize(games: list[dict]) -> dict:
    contacts = [g["first_contact"] for g in games if g["first_contact"] is not None]
    wars = [g["first_war"] for g in games if g["first_war"] is not None]
    low, high = TARGETS["first_contact"]
    w_low, w_high = TARGETS["war_pairs"]

    by_name: dict[str, dict] = defaultdict(lambda: {"points": [], "wins": 0, "attacks": 0,
                                                    "actions": Counter()})
    for game in games:
        for seat in game["per_seat"]:
            entry = by_name[seat["name"]]
            entry["points"].append(seat["points"])
            entry["wins"] += seat["rank"] == 0
            entry["attacks"] += seat["attacks"]
            entry["actions"].update(seat["actions"])

    strategies = {}
    for name, entry in sorted(by_name.items()):
        total = sum(entry["actions"].values()) or 1
        strategies[name] = {
            "seats": len(entry["points"]),
            "mean_points": statistics.mean(entry["points"]),
            "win_rate": entry["wins"] / len(entry["points"]),
            "attacks_per_game": entry["attacks"] / len(entry["points"]),
            "action_mix": {
                action: count / total for action, count in entry["actions"].most_common()
            },
        }

    return {
        "games": len(games),
        "first_contact": {
            "median": statistics.median(contacts) if contacts else None,
            "mean": statistics.mean(contacts) if contacts else None,
            "min": min(contacts, default=None),
            "max": max(contacts, default=None),
            "never": _share(g["first_contact"] is None for g in games),
            "in_target": _share(
                g["first_contact"] is not None and low <= g["first_contact"] <= high
                for g in games
            ),
        },
        "first_war": {
            "median": statistics.median(wars) if wars else None,
            "never": _share(g["first_war"] is None for g in games),
        },
        "war_pairs": {
            "mean": statistics.mean(g["war_pairs"] for g in games),
            "distribution": dict(sorted(Counter(g["war_pairs"] for g in games).items())),
            "in_target": _share(w_low <= g["war_pairs"] <= w_high for g in games),
        },
        "leader_mid": {
            "win_rate": _share(g["leader_mid_won"] for g in games),
            "spearman": statistics.mean(g["mid_final_spearman"] for g in games),
        },
        "battles_per_game": statistics.mean(g["battles"] for g in games),
        "eliminations_per_game": statistics.mean(g["eliminations"] for g in games),
        "strategies": strategies,
    }


# ── rendu ───────────────────────────────────────────────────────────────────

def _verdict(value, low, high, below="trop bas", above="trop haut") -> str:
    if value is None:
        return "jamais"
    return "OK" if low <= value <= high else (below if value < low else above)


def render(summary: dict, mid_turn: int) -> str:
    fc, war, wp, lm = (summary[k] for k in ("first_contact", "first_war", "war_pairs",
                                             "leader_mid"))
    low, high = TARGETS["first_contact"]
    l_low, l_high = TARGETS["leader_mid_wins"]
    w_low, w_high = TARGETS["war_pairs"]
    lines = [
        f"── Dynamique · {summary['games']} parties ──────────────────────────────",
        "",
        f"Premier contact   médiane T{fc['median']}  (min T{fc['min']}, max T{fc['max']})"
        f"   cible T{low:g}–T{high:g} : {_verdict(fc['median'], low, high, 'trop tôt', 'trop tard')}"
        f"   · {fc['in_target']:.0%} des parties dans la cible"
        + (f" · jamais : {fc['never']:.0%}" if fc["never"] else ""),
        f"Première guerre   médiane T{war['median']}"
        + (f" · jamais : {war['never']:.0%}" if war["never"] else ""),
        f"Paires en guerre  moyenne {wp['mean']:.2f} / 6   répartition {wp['distribution']}"
        f"   cible {w_low}–{w_high} : {wp['in_target']:.0%} des parties",
        f"Leader au T{mid_turn}      finit 1er dans {lm['win_rate']:.0%} des parties"
        f"   cible {l_low:.0%}–{l_high:.0%} : {_verdict(lm['win_rate'], l_low, l_high)}"
        f"   · corrélation score T{mid_turn} / rang final {lm['spearman']:.2f}",
        f"Batailles         {summary['battles_per_game']:.1f} par partie"
        f" · éliminations {summary['eliminations_per_game']:.2f} par partie",
        "",
        f"{'stratégie':<20} {'sièges':>6} {'pts rang':>8} {'victoires':>9} {'attaques':>8}   actions principales",
    ]
    for name, s in summary["strategies"].items():
        mix = ", ".join(f"{a} {share:.0%}" for a, share in list(s["action_mix"].items())[:4])
        lines.append(
            f"{name:<20} {s['seats']:>6} {s['mean_points']:>8.2f} {s['win_rate']:>9.0%}"
            f" {s['attacks_per_game']:>8.1f}   {mix}"
        )
    lines.append("")
    lines.append("pts rang : 1er = 3, 2e = 2, 3e = 1, 4e = 0 → 1.50 = moyenne neutre · "
                 "attaques : offensives lancées par partie")
    return "\n".join(lines)


# ── ligne de commande ───────────────────────────────────────────────────────

def _seeds(text: str) -> list[int]:
    if "-" in text:
        start, end = text.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in text.split(",")]


def _overrides(pairs: list[str]) -> dict | None:
    """`world.grid_size=10` → `{"world": {"grid_size": 10}}`."""
    if not pairs:
        return None
    out: dict = {}
    for pair in pairs:
        dotted, raw = pair.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = out
        *path, leaf = dotted.split(".")
        for key in path:
            node = node.setdefault(key, {})
        node[leaf] = value
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lab.dynamics", description=__doc__.split("\n")[0])
    parser.add_argument("--panel", default=",".join(DEFAULT_PANEL))
    parser.add_argument("--seeds", default="1-12", help="ex. 1-12 ou 1,4,7")
    parser.add_argument("--mid", type=int, default=7, help="tour de mi-partie (défaut 7)")
    parser.add_argument("--set", action="append", default=[], metavar="CLÉ=VALEUR",
                        help="constante modifiée, ex. world.grid_size=10 (répétable)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default=None, help="JSON (défaut : runs/lab/dynamics-<date>.json)")
    args = parser.parse_args(argv)

    panel = [name.strip() for name in args.panel.split(",")]
    overrides = _overrides(args.set)
    started = time.perf_counter()
    games = run(panel, _seeds(args.seeds), overrides, args.mid, args.workers,
                on_game=lambda done, total: print(f"\r{done}/{total} parties", end="", flush=True))
    print()
    summary = summarize(games)
    print(render(summary, args.mid))

    out = Path(args.out or f"runs/lab/dynamics-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "panel": panel, "seeds": args.seeds, "mid_turn": args.mid, "overrides": overrides,
        "targets": TARGETS, "summary": summary, "games": games,
        "seconds": round(time.perf_counter() - started, 1),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(games)} parties en {time.perf_counter() - started:.0f} s → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
