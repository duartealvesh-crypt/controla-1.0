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
    from controla.policies.factory import available_seats, build_policies
    from controla.replay.render import render_match

    seats = [name.strip() for name in args.policies.split(",")]
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


def cmd_replay(args: argparse.Namespace) -> int:
    """Rejoue une partie stockée, à seed fixée."""
    from controla.replay.render import render_match

    result = json.loads(Path(args.path).read_text(encoding="utf-8"))
    print(render_match(result))
    return 0


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

    replay = subparsers.add_parser("replay", help="rejoue une partie stockée")
    replay.add_argument("path", help="fichier JSON produit par `play --out`")
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
