"""Rendu texte : état courant et replay d'une partie.

Un replay lisible est un livrable à part entière (§14.6) : à seed fixée on doit
pouvoir relire une partie et comprendre POURQUOI elle a tourné ainsi. Chaque
événement porte donc sa cause.
"""

from __future__ import annotations

from controla.dynamics.scoring import score
from controla.registry.catalog import Catalog
from controla.state.game_state import GameState

# Un caractère par terrain, un par bâtiment. Purement cosmétique.
_TERRAIN_GLYPHS = {"plain": ".", "river": "~", "forest": "#"}
_BUILDING_GLYPHS = {"city": "C", "lab": "L", "camp": "X", "field": "F"}
_OWNER_GLYPHS = "0123"


def render_map(state: GameState) -> str:
    """La carte. Majuscule = bâtiment, chiffre = propriétaire, sinon terrain."""
    lines = []
    for y in range(state.grid.size):
        row = []
        for x in range(state.grid.size):
            tile = state.grid.get(x, y)
            if tile.building is not None:
                row.append(_BUILDING_GLYPHS.get(tile.building, "?"))
            elif tile.owner is not None:
                row.append(_OWNER_GLYPHS[tile.owner % len(_OWNER_GLYPHS)])
            else:
                row.append(_TERRAIN_GLYPHS.get(tile.terrain, "?"))
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
