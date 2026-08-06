"""Rendu texte : état courant et replay d'une partie.

Un replay lisible est un livrable à part entière (§14.6) : à seed fixée on doit
pouvoir relire une partie et comprendre POURQUOI elle a tourné ainsi. Chaque
événement porte donc sa cause.
"""

from __future__ import annotations

from controla.dynamics.scoring import score
from controla.registry.catalog import Catalog
from controla.state.game_state import GameState

# Un caractère par terrain, un par bâtiment. Cosmétique, mais PAS arbitraire :
# les glyphes doivent rester deux à deux distincts, sans quoi deux bâtiments
# différents deviennent indiscernables à l'écran. « Camp militaire » et
# « Champs » commencent tous deux par C — les dériver du libellé produirait
# exactement cette collision, d'où cette table explicite.
#
# Publiques parce que l'interface web les sert au navigateur : une seconde table
# côté JavaScript rejouerait la collision qu'on vient d'éviter.
TERRAIN_GLYPHS = {"plain": ".", "river": "~", "forest": "#"}
BUILDING_GLYPHS = {"city": "C", "lab": "L", "camp": "X", "field": "F"}
_OWNER_GLYPHS = "0123"


def building_glyph(key: str) -> str:
    """Glyphe d'un bâtiment. Un mod non déclaré retombe sur son initiale — au
    risque d'une collision, que la table ci-dessus est justement là pour lever."""
    return BUILDING_GLYPHS.get(key, key[:1].upper() or "?")


def render_map(state: GameState) -> str:
    """La carte. Majuscule = bâtiment, chiffre = propriétaire, sinon terrain."""
    lines = []
    for y in range(state.grid.size):
        row = []
        for x in range(state.grid.size):
            tile = state.grid.get(x, y)
            if tile.building is not None:
                row.append(building_glyph(tile.building))
            elif tile.owner is not None:
                row.append(_OWNER_GLYPHS[tile.owner % len(_OWNER_GLYPHS)])
            else:
                row.append(TERRAIN_GLYPHS.get(tile.terrain, "?"))
        lines.append(" ".join(row))
    return "\n".join(lines)


def render_state(cfg, catalog: Catalog, state: GameState) -> str:
    """L'état complet, du point de vue de l'arbitre (aucun brouillard)."""
    header = f"── Tour {state.turn}/{cfg.world.max_turns} " + "─" * 40
    rows = [header, render_map(state), ""]
    for faction in state.factions:
        if not faction.alive:
            rows.append(f"  {faction.name:10s} ÉLIMINÉ au tour {faction.eliminated_turn}")
            continue
        buildings = state.building_counts(faction.id)
        traits = " ".join(
            f"{key[:4]}{value:+.2f}" for key, value in faction.traits.items()
        )
        rows.append(
            f"  {faction.name:10s} "
            f"pop {faction.population:6.1f}  "
            f"nour {faction.food:6.1f}  bois {faction.wood:6.1f}  "
            f"mét {faction.metal:5.1f}  "
            f"rech {faction.research:5.1f}(T{faction.research_tier})  "
            f"force {faction.force:7.1f}  "
            f"cases {state.tile_count(faction.id):3d}  "
            f"bât {dict(sorted(buildings.items()))}  "
            f"score {score(cfg, state, faction.id):7.1f}"
        )
        rows.append(f"             traits {traits}   faim {faction.hunger:.2f}")
    return "\n".join(rows)


def render_turn(report: dict, names: dict[int, str]) -> str:
    """Un tour de replay : ce qui s'est passé, et pourquoi."""
    def who(faction_id) -> str:
        return names.get(faction_id, f"F{faction_id}")

    lines = [f"═══ Tour {report['turn']} " + "═" * 50]

    for event in report["events"]:
        kind = event.get("kind")
        actor = who(event.get("faction")) if event.get("faction") is not None else "?"
        cause = event.get("cause", "")

        if kind == "divine_action":
            target = event.get("target")
            suffix = "" if target is None else f" → {target}"
            effects = ", ".join(item["effect"] for item in event["effects"])
            lines.append(f"  ✦ {actor} : {event['action']}{suffix}  [{effects}]")
        elif kind == "expansion":
            lines.append(f"  ▸ {actor} s'étend en {tuple(event['tile'])}  — {cause}")
        elif kind == "build":
            lines.append(
                f"  ⌂ {actor} bâtit {event['building']} en {tuple(event['tile'])}  — {cause}"
            )
        elif kind == "trade":
            pair = " ↔ ".join(who(f) for f in event["factions"])
            lines.append(f"  ⇄ Commerce {pair} : {event['moved']}  — {cause}")
        elif kind == "war_launched":
            surprise = "  ⚡SURPRISE" if event["surprise"] else ""
            lines.append(
                f"  ⚔ {actor} lance l'assaut sur {who(event['target'])} "
                f"en {tuple(event['tile'])}{surprise}  — {cause}"
            )
        elif kind == "research_tier":
            lines.append(f"  ✧ {actor} atteint le palier de recherche {event['tier']}")
        elif kind in ("disease", "natural_disaster", "discovery"):
            lines.append(f"  🌍 {actor} : {cause}")
        elif kind == "elimination":
            lines.append(f"  ☠ {actor} est éliminé — {cause}")
        elif kind == "drift":
            moved = ", ".join(f"{k}{v:+.3f}" for k, v in event["moved"].items())
            lines.append(f"  ↝ {actor} dérive : {moved}")

    for battle in report["battles"]:
        verdict = "VICTOIRE" if battle["attacker_won"] else "DÉFAITE"
        allies = (
            f" (+ {', '.join(who(a) for a in battle['allies'])})"
            if battle["allies"] else ""
        )
        lines.append(
            f"  ⚔ Bataille {who(battle['attacker'])}{allies} vs "
            f"{who(battle['defender'])} : {verdict} "
            f"(p={battle['win_probability']:.2f}, "
            f"{battle['attacker_force']:.0f} vs {battle['defender_force']:.0f})"
        )
        lines.append(
            f"      pertes {battle['attacker_losses']:.1f} / "
            f"{battle['defender_losses']:.1f}"
            + (f", annexe {tuple(battle['annexed'])}" if battle["annexed"] else "")
            + (f", butin {battle['loot']}" if battle["loot"] else "")
        )

    return "\n".join(lines)


def render_match(result: dict) -> str:
    """Le replay complet d'une partie, depuis son résultat stocké."""
    names = {
        entry["faction_id"]: entry["name"] for entry in result["ranking"]
    }
    blocks = [
        f"Partie seed={result['seed']}  sièges={result['seats']}",
        "",
    ]
    for report in result["history"]:
        rendered = render_turn(report, names)
        if rendered.count("\n") > 0:
            blocks.append(rendered)
    blocks.append("")
    blocks.append("═══ CLASSEMENT FINAL " + "═" * 45)
    for entry in result["ranking"]:
        status = "" if entry["alive"] else f"  (éliminé T{entry['eliminated_turn']})"
        reputation = entry["reputation"]
        trust = (
            f"  parole {reputation['honored']}/{reputation['total']}"
            if reputation.get("total") else ""
        )
        blocks.append(
            f"  {entry['rank'] + 1}. {entry['name']:10s} "
            f"score {entry['score']:8.1f}  {entry['components']}{status}{trust}"
        )
    return "\n".join(blocks)


def _cell(value, digits: int = 2, width: int = 6) -> str:
    """Une valeur numérique, ou `—` quand la métrique n'a pas de sens ici."""
    return f"{'—':>{width}}" if value is None else f"{value:>{width}.{digits}f}"


def render_leaderboard(manifest: dict) -> str:
    """Le classement d'une campagne et les cinq métriques §12.

    L'ordinal `μ − 3σ` fait foi ; `μ` et `σ` restent affichés parce qu'un ordinal
    bas peut vouloir dire « faible » comme « pas encore assez observé », et que ces
    deux lectures appellent des suites différentes.
    """
    spec = manifest.get("spec", {})
    metrics = manifest.get("metrics", {})
    lines = [
        f"Campagne {manifest.get('run', '?')} — {manifest.get('games', 0)} parties "
        f"({len(spec.get('seeds', []))} seeds × {spec.get('rotations', 0)} rotations "
        f"× {spec.get('repetitions', 1)} rép.)  {manifest.get('wall_seconds', 0)} s",
        "",
        "  # participant        μ−3σ      μ      σ  parties  vict.  rang",
        "  " + "─" * 62,
    ]
    for position, entry in enumerate(manifest.get("leaderboard", []), start=1):
        lines.append(
            f"  {position}. {entry['name']:16s}"
            f"{_cell(entry['ordinal'])} {_cell(entry['mu'])} {_cell(entry['sigma'])}"
            f"  {entry['games']:6d} {entry['wins']:6d} {_cell(entry['mean_rank'])}"
        )

    lines += [
        "",
        "  participant       parole  exact.  focus  engag.  allian.  survie",
        "  " + "─" * 62,
    ]
    for name, report in metrics.items():
        coherence, alliances = report["coherence"], report["alliances"]
        lines.append(
            f"  {name:16s}"
            f"{_cell(report['parole']['rate'])}"
            f"{_cell(report['exactitude']['rate'], width=8)}"
            f"{_cell(coherence['focus'], width=7)}"
            f"{_cell(coherence['engagement'], width=8)}"
            f"{_cell(alliances['formed_per_game'], width=9)}"
            f"{_cell(report['survie']['survival_rate'], width=8)}"
        )
    if "criteria" in manifest:
        lines += ["", render_criteria(manifest["criteria"])]
    return "\n".join(lines)


def _verdict(value) -> str:
    """`—` quand le critère ne s'évalue pas sur ce panel, et non un faux échec."""
    return "—" if value is None else ("OK" if value else "NON")


def render_criteria(criteria: dict) -> str:
    """Les trois critères de sortie de §13, en une ligne chacun."""
    horizon, balance, skill = (
        criteria["horizon"], criteria["balance"], criteria["skill"]
    )
    verdict = criteria["verdict"]
    return "\n".join([
        "  §13 (a) horizon    "
        f"{_verdict(verdict['a_horizon']):>4s}   tour moyen {horizon['mean_turns']}, "
        f"{_cell(horizon['reached_max'], 2, 4)} au bout, "
        f"{horizon['eliminations_per_game']} élimination/partie",
        "  §13 (b) équilibre  "
        f"{_verdict(verdict['b_balance']):>4s}   domination "
        f"{_cell(balance['domination'], 2, 4)}, plancher "
        f"{_cell(balance['floor'], 2, 4)}, étalement "
        f"{_cell(balance['spread'], 2, 4)}",
        "  §13 (c) skill      "
        f"{_verdict(verdict['c_skill']):>4s}   écart au hasard "
        f"{_cell(skill['over_chance'], 1, 6)}, "
        f"amplitude du panel {_cell(skill['separation'], 1, 6)}",
    ])


def render_sweep(manifest: dict) -> str:
    """Le tableau comparatif d'une calibration : une ligne par variante et panel."""
    lines = [
        f"Sweep « {manifest['sweep']} » — {manifest['cells']} variantes, "
        f"{manifest['games']} parties, {manifest['wall_seconds']} s",
        "",
        f"  {'variante':26s} {'panel':12s} {'domin.':>7s} {'planch.':>8s} "
        f"{'étalem.':>8s} {'hasard':>7s} {'abc':>5s}  meneur",
        "  " + "─" * 92,
    ]
    for row in manifest["rows"]:
        criteria = row["criteria"]
        verdicts = criteria["verdict"]
        flags = "".join(
            "." if verdicts[key] is None else ("o" if verdicts[key] else "X")
            for key in ("a_horizon", "b_balance", "c_skill")
        )
        leader = row["leaderboard"][0]
        lines.append(
            f"  {row['cell']:26.26s} {row['panel']:12.12s} "
            f"{_cell(criteria['balance']['domination'], 2, 7)}"
            f"{_cell(criteria['balance']['floor'], 2, 8)}"
            f"{_cell(criteria['balance']['spread'], 2, 8)}"
            f"{_cell(criteria['skill']['over_chance'], 1, 7)}"
            f"{flags:>5s}  {leader['name']} ({leader['wins']}v)"
        )
    lines += [
        "",
        "  abc = les trois critères §13 : o atteint · X manqué · . non évalué",
    ]
    return "\n".join(lines)
