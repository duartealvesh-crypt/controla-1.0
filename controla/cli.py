"""Points d'entrée en ligne de commande.

Les sous-commandes arrivent avec leur tranche : `describe` (T0), `play` (T1),
`tournament` (T4), `sweep` (T5), `replay` (T6).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from controla.config.loader import load_config
from controla.registry.catalog import build_catalog
from controla.rules_text import render_rules


def cmd_describe(args: argparse.Namespace) -> int:
    """Imprime le règlement complet, dérivé des seuls descripteurs.

    Sert de contrôle visuel : ce que le dieu lira dans son prompt est exactement
    ce que le moteur applique.
    """
    cfg = load_config(args.config)
    catalog = build_catalog(cfg)
    print(render_rules(cfg, catalog))
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    """Joue une partie et affiche son replay."""
    from controla.arena.play import MatchSpec, play_match
    from controla.policies.factory import build_policies
    from controla.replay.render import render_match

    seats = _known_seats(args.policies)
    if seats is None:
        return 2

    cfg = load_config(args.config)
    catalog = build_catalog(cfg)
    result = play_match(
        MatchSpec(seed=args.seed, seats=seats, journal_rng=args.journal),
        build_policies(seats, cfg, catalog),
        config=cfg,
    )
    for agent, report in result.get("policy_reports", {}).items():
        print(f"[{agent}] {report}", file=sys.stderr)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"partie écrite dans {args.out}", file=sys.stderr)

    print(render_match(result) if not args.json
          else json.dumps(result, ensure_ascii=False, indent=1))
    return 0


def cmd_tournament(args: argparse.Namespace) -> int:
    """Joue une campagne : rotations de sièges × seeds × répétitions."""
    from controla.arena.tournament import TournamentSpec, run_tournament
    from controla.replay.render import render_leaderboard

    seats = _known_seats(args.policies)
    if seats is None:
        return 2

    spec = TournamentSpec(
        seats=seats,
        seeds=[int(value) for value in args.seeds.split(",")],
        rotations=args.rotations,
        repetitions=args.reps,
        run_dir=Path(args.out) if args.out else None,
        config_path=args.config,
    )

    def progress(index: int, total: int, result: dict) -> None:
        winner = result["seats"][result["ranking"][0]["faction_id"]]
        print(
            f"  [{index + 1:>4}/{total}] seed {result['seed']:<5} "
            f"{','.join(result['seats'])} → {winner}",
            file=sys.stderr,
        )

    manifest = run_tournament(spec, config=load_config(args.config), on_game=progress)
    print(render_leaderboard(manifest) if not args.json
          else json.dumps(manifest, ensure_ascii=False, indent=1))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Joue un plan de calibration : une variante de constante à la fois."""
    from controla.arena.sweep import load_sweep, run_sweep
    from controla.replay.render import render_sweep

    spec = load_sweep(args.path, run_dir=args.out)
    print(
        f"plan « {spec.name} » : {len(spec.cells())} variantes × "
        f"{len(spec.panels)} panels = {spec.games()} parties",
        file=sys.stderr,
    )
    if args.dry_run:
        for cell in spec.cells():
            print(f"  {cell.label:30s} {cell.overrides}", file=sys.stderr)
        return 0

    def progress(step: int, total: int, label: str, panel: str, manifest: dict) -> None:
        criteria = manifest["criteria"]["balance"]
        print(
            f"  [{step:>3}/{total}] {label:28.28s} {panel:12.12s} "
            f"domination {criteria['domination']:.0%}",
            file=sys.stderr,
        )

    manifest = run_sweep(spec, on_cell=progress)
    print(render_sweep(manifest) if not args.json
          else json.dumps(manifest, ensure_ascii=False, indent=1))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Rejoue une partie stockée, à seed fixée."""
    from controla.arena.storage import load_match
    from controla.replay.render import render_match

    print(render_match(load_match(args.path)))
    return 0


def _known_seats(raw: str) -> list[str] | None:
    """Découpe et valide une liste de sièges. `None` si l'un est inconnu."""
    from controla.policies.factory import available_seats

    seats = [name.strip() for name in raw.split(",")]
    catalogue = available_seats()
    known = set(catalogue["scripted"]) | set(catalogue["models"])
    unknown = [name for name in seats if name not in known]
    if unknown:
        print(
            f"sièges inconnus : {unknown}\n"
            f"  archétypes : {catalogue['scripted']}\n"
            f"  modèles    : {catalogue['models']}",
            file=sys.stderr,
        )
        return None
    return seats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="controla",
        description="Arène god-4X pour benchmark de LLM.",
    )
    parser.add_argument(
        "--config", default=None,
        help="YAML de constantes (défaut : controla/config/default.yaml)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser(
        "describe", help="imprime le règlement dérivé des descripteurs",
    )
    describe.set_defaults(func=cmd_describe)

    play = subparsers.add_parser("play", help="joue une partie et affiche son replay")
    play.add_argument("--seed", type=int, default=42)
    play.add_argument(
        "--policies", default="cooperator,betrayer,expansionist,turtle",
        help="une policy par siège, séparées par des virgules",
    )
    play.add_argument("--out", default=None, help="écrit le résultat en JSON")
    play.add_argument("--json", action="store_true", help="sortie JSON plutôt que replay")
    play.add_argument(
        "--journal", action="store_true",
        help="journalise chaque tirage d'aléa (audit et empreinte de partie)",
    )
    play.set_defaults(func=cmd_play)

    tournament = subparsers.add_parser(
        "tournament", help="joue une campagne et classe les participants",
    )
    tournament.add_argument(
        "--policies", "--models", dest="policies",
        default="cooperator,betrayer,expansionist,turtle",
        help="un participant par siège, séparés par des virgules",
    )
    tournament.add_argument(
        "--seeds", default="1,2,3", help="seeds de monde, séparées par des virgules",
    )
    tournament.add_argument(
        "--rotations", type=int, default=0,
        help="rotations de sièges par seed (0 = toutes : chacun joue chaque coin)",
    )
    tournament.add_argument(
        "--reps", type=int, default=1,
        help="répétitions par alignement — utile aux policies stochastiques (LLM) "
             "seulement : à monde et sièges fixés, une policy scriptée rejoue "
             "exactement la même partie",
    )
    tournament.add_argument(
        "--out", default=None, help="dossier du run (défaut : runs/<horodatage>)",
    )
    tournament.add_argument(
        "--json", action="store_true", help="manifeste JSON plutôt que tableau",
    )
    tournament.set_defaults(func=cmd_tournament)

    sweep = subparsers.add_parser(
        "sweep", help="calibre : une variante de constante à la fois, zéro appel LLM",
    )
    sweep.add_argument("path", help="plan de calibration YAML (voir configs/)")
    sweep.add_argument(
        "--out", default=None, help="dossier du sweep (défaut : runs/sweep-<nom>)",
    )
    sweep.add_argument(
        "--dry-run", action="store_true",
        help="liste les variantes et le nombre de parties, sans rien jouer",
    )
    sweep.add_argument("--json", action="store_true", help="manifeste JSON")
    sweep.set_defaults(func=cmd_sweep)

    replay = subparsers.add_parser("replay", help="rejoue une partie stockée")
    replay.add_argument(
        "path", help="JSON de `play --out` ou JSONL de `tournament`",
    )
    replay.set_defaults(func=cmd_replay)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Le règlement contient des flèches et des filets Unicode ; la console Windows
    # est en cp1252 par défaut et lèverait UnicodeEncodeError. On force UTF-8 plutôt
    # que d'appauvrir le texte lu par le dieu.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
