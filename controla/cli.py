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
        MatchSpec(
            seed=args.seed, seats=seats, journal_rng=args.journal,
            snapshots=args.snapshots,
        ),
        build_policies(seats, cfg, catalog),
        config=cfg,
    )
    from controla.replay.render import render_llm_usage

    usage = render_llm_usage(result)
    if usage:
        print(usage, file=sys.stderr)

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


def cmd_models(args: argparse.Namespace) -> int:
    """Confronte le registre de modèles au catalogue réel du fournisseur.

    C'est la commande à passer AVANT une campagne : un identifiant périmé ou un
    réglage d'échantillonnage refusé se paie sinon en parties jouées pour rien,
    ou pire en classement faussé (cf. `policies/llm/catalogue.py`).
    """
    from controla.policies.llm.catalogue import (
        audit,
        campaign_cost,
        load_provider_catalogues,
    )
    from controla.policies.llm.registry import load_dotenv_if_present, load_models

    found = load_dotenv_if_present()
    print(f"# .env : {found or 'aucun trouvé'}", file=sys.stderr)

    declared = load_models()
    if not declared:
        print("aucun modèle déclaré dans policies/llm/models.yaml", file=sys.stderr)
        return 1

    width = max(len(name) for name in declared) + 2
    if not args.check:
        for name, entry in sorted(declared.items()):
            print(f"  {name:{width}s} {entry.get('provider', '?'):12s} {_flags(entry)}")
        return 0

    catalogues, errors = load_provider_catalogues(declared)
    for provider, message in sorted(errors.items()):
        print(f"  ✗ {provider:14s} {message}", file=sys.stderr)

    reports = audit(declared, catalogues)
    header = (
        f"  {'siège':{width}s} {'$/Mtok in':>10s} {'$/Mtok out':>11s} "
        f"{f'{args.games} parties':>11s}  options"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    failures = 0
    for report in reports:
        cost = campaign_cost(report, games=args.games)
        print(
            f"  {report['model']:{width}s} "
            f"{_money(report.get('price_in')):>10s} "
            f"{_money(report.get('price_out')):>11s} "
            f"{_money(cost, prefix='$'):>11s}  "
            f"{_flags(declared.get(report['model'], {}))}"
        )
        for problem in report["problems"]:
            failures += 1
            print(f"      ✗ {problem}", file=sys.stderr)
        for warning in report["warnings"]:
            print(f"      ~ {warning}", file=sys.stderr)

    print(
        f"\n  Coût extrapolé pour UN siège sur {args.games} parties, à partir des "
        "45 000 tokens/siège/partie mesurés en T2 (docs/FINDINGS.md n° 5).\n"
        "  Il ignore le cache de prompt et la verbosité propre au modèle.",
        file=sys.stderr,
    )
    return 1 if failures or errors else 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Mesure des modèles candidats sur les vrais prompts, au premier essai."""
    import time as _time

    from controla.policies.llm import bench
    from controla.policies.llm.registry import load_dotenv_if_present, load_models

    load_dotenv_if_present()
    cfg = load_config(args.config)
    catalog = build_catalog(cfg)

    if args.models:
        models = [name.strip() for name in args.models.split(",") if name.strip()]
    else:
        print("# découverte des modèles gratuits (données publiques OpenRouter)…",
              file=sys.stderr)
        found = bench.discover_free_candidates(min_uptime=args.min_uptime)
        for entry in found:
            uptime = "—" if entry["uptime_1d"] is None else f"{entry['uptime_1d']:.0f} %"
            print(f"  {entry['id']:56s} dispo 24 h {uptime:>6s}  "
                  f"sortie max {entry['max_completion_tokens'] or '—'}", file=sys.stderr)
        models = [entry["id"] for entry in found]
    if not models:
        print("aucun modèle à mesurer", file=sys.stderr)
        return 1

    probes = bench.build_probes(
        cfg, catalog,
        seeds=[int(s) for s in args.seeds.split(",")],
        turns=[int(t) for t in args.turns.split(",")],
    )
    total = bench.plan_size(models, probes)
    minutes = total / args.rpm if args.rpm else 0
    print(
        f"\nbanc : {len(models)} modèles × {len(probes)} sondes = {total} requêtes, "
        f"~{minutes:.0f} min au rythme de {args.rpm:g}/min.\n"
        f"Quota gratuit OpenRouter : {bench.FREE_REQUESTS_PER_DAY}/jour sans crédit "
        "acheté, 1 000/jour après 10 $ de crédits.",
        file=sys.stderr,
    )
    if args.dry_run:
        for probe in probes:
            print(f"  sonde {probe.label:22s} {probe.points} pts, "
                  f"{len(probe.costs)} coups légaux", file=sys.stderr)
        return 0

    declared = load_models()
    backends = {name: bench.make_backend(name, cfg, declared.get(name)) for name in models}

    def progress(name, probe, call):
        state = ("✓" if call["well_formed"] else "✗") + (f" {call['error']}" if call["error"] else "")
        print(f"  {name:56.56s} {probe.label:20s} {call['latency']:6.1f} s  {state}",
              file=sys.stderr)

    summaries = bench.rank(bench.run_bench(backends, probes, args.rpm, on_call=progress))

    out = Path(args.out or f"runs/bench/{_time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "models": models, "probes": [p.label for p in probes],
        "max_tokens": cfg.llm.max_tokens, "results": summaries,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    pct = lambda v: "—" if v is None else f"{v:.0%}"          # noqa: E731
    sec = lambda v: "—" if v is None else f"{v:.1f} s"        # noqa: E731
    width = max(len(s["model"]) for s in summaries) + 2
    print(f"\n  {'modèle':{width}s} {'répond':>7s} {'forme':>6s} {'légal':>6s} "
          f"{'budget':>7s} {'latence':>8s} {'max':>7s} {'sortie':>7s}  erreurs")
    for s in summaries:
        errors = ", ".join(f"{k}×{v}" for k, v in s["errors"].items()) or "—"
        print(f"  {s['model']:{width}s} {pct(s['answered']):>7s} {pct(s['well_formed']):>6s} "
              f"{pct(s['legal_rate']):>6s} {pct(s['within_budget']):>7s} "
              f"{sec(s['latency_median']):>8s} {sec(s['latency_max']):>7s} "
              f"{s['completion_tokens']:>7d}  {errors}")
    print(f"\n  détail par appel : {out}", file=sys.stderr)
    return 0


def _flags(entry: dict) -> str:
    """Les options qui changent le TRAITEMENT d'un siège, à voir d'un coup d'œil."""
    flags = []
    if entry.get("model"):
        flags.append(f"→ {entry['model']}")
    if entry.get("memory"):
        flags.append("mémoire")
    if entry.get("cache_control"):
        flags.append("cache explicite")
    if entry.get("reasoning_effort"):
        flags.append(f"réflexion {entry['reasoning_effort']}")
    return "  ".join(flags)


def _money(value, prefix: str = "") -> str:
    """`—` quand le fournisseur ne publie pas de prix — et non `0.000`, qui
    ferait passer un tarif inconnu pour la gratuité."""
    return "—" if value is None else f"{prefix}{value:,.3f}"


def cmd_ui(args: argparse.Namespace) -> int:
    """Lance l'interface web locale : replay, campagnes, partie jouable."""
    try:
        import uvicorn
    except ImportError:                                # pragma: no cover
        print(
            "l'interface web demande fastapi et uvicorn :\n"
            "  pip install fastapi 'uvicorn[standard]'",
            file=sys.stderr,
        )
        return 1

    from controla.ui.app import create_app

    print(
        f"Controla — http://{args.host}:{args.port}  (racine : {args.runs})\n"
        "Serveur LOCAL et sans authentification : ne l'exposez pas.",
        file=sys.stderr,
    )
    uvicorn.run(
        create_app(root=args.runs, config_path=args.config),
        host=args.host, port=args.port, log_level="warning",
    )
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
    play.add_argument(
        "--snapshots", action="store_true",
        help="joint la carte et l'état du plateau à chaque tour, ce qui rend la "
             "partie rejouable dans l'interface web (`cli ui`)",
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

    models = subparsers.add_parser(
        "models", help="liste et vérifie le registre de modèles LLM",
    )
    models.add_argument(
        "--check", action="store_true",
        help="interroge le fournisseur : le modèle existe-t-il, à quel prix, et "
             "accepte-t-il les réglages déclarés ? (code de sortie 1 si non)",
    )
    models.add_argument(
        "--games", type=int, default=48,
        help="nombre de parties pour l'estimation de coût (défaut : 48)",
    )
    models.set_defaults(func=cmd_models)

    bench_parser = subparsers.add_parser(
        "bench",
        help="mesure des modèles sur les vrais prompts : forme, légalité, budget, latence",
    )
    bench_parser.add_argument(
        "--models", default=None,
        help="identifiants à mesurer, séparés par des virgules (défaut : découverte "
             "automatique des modèles gratuits jouables d'OpenRouter)",
    )
    bench_parser.add_argument("--seeds", default="1", help="seeds des parties sondes")
    bench_parser.add_argument(
        "--turns", default="0,6,12", help="tours sondés (début, milieu, fin de partie)",
    )
    bench_parser.add_argument(
        "--min-uptime", type=float, default=90.0,
        help="disponibilité 24 h minimale pour la découverte, en %% (défaut : 90)",
    )
    bench_parser.add_argument(
        "--rpm", type=float, default=20.0,
        help="requêtes par minute (palier gratuit OpenRouter : 20)",
    )
    bench_parser.add_argument("--out", default=None, help="JSON de résultats")
    bench_parser.add_argument(
        "--dry-run", action="store_true",
        help="annonce les modèles, les sondes et le nombre de requêtes, sans appel",
    )
    bench_parser.set_defaults(func=cmd_bench)

    ui = subparsers.add_parser(
        "ui", help="interface web locale : replay, campagnes, partie jouable",
    )
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8000)
    ui.add_argument(
        "--runs", default="runs",
        help="racine servie — AUCUN chemin hors de ce dossier n'est lisible",
    )
    ui.set_defaults(func=cmd_ui)

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
