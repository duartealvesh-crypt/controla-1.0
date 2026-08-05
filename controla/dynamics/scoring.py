"""Score, classement, et fin de partie (spec §10).

Le score est **multi-axes volontairement** : plusieurs voies de victoire (éco,
militaire, tech) restent viables, donc aucun build unique ne domine et
l'intransitivité est préservée.

Le classement complet des 4 dieux est ce dont OpenSkill / Plackett-Luce a besoin.
Il n'y a jamais de match nul : les égalités sont brisées de façon déterministe.
"""

from __future__ import annotations

from controla.state.game_state import GameState


def score_components(state: GameState, faction_id: int) -> dict[str, float]:
    """Les axes du score, séparés — utile au diagnostic (quelle voie a payé ?)."""
    faction = state.faction(faction_id)
    return {
        "population": faction.population,
        "tiles": float(state.tile_count(faction_id)),
        "research": faction.research,
        "buildings": float(state.building_total(faction_id)),
    }


def score(cfg, state: GameState, faction_id: int) -> float:
    weights = cfg.score.weights
    return sum(
        value * float(weights[axis])
        for axis, value in score_components(state, faction_id).items()
    )


def ranking(cfg, state: GameState) -> list[int]:
    """Les IDs de faction, du meilleur au pire.

    Les survivants passent devant les éliminés ; entre éliminés, mourir tard vaut
    mieux que mourir tôt (c'est une vraie information de performance). L'ID sert
    de dernier départage, pour qu'un replay produise toujours le même ordre.
    """
    def sort_key(faction_id: int) -> tuple:
        faction = state.faction(faction_id)
        return (
            0 if faction.alive else 1,          # survivants d'abord
            -score(cfg, state, faction_id) if faction.alive else 0,
            -faction.eliminated_turn if not faction.alive else 0,
            faction_id,
        )

    return sorted(range(len(state.factions)), key=sort_key)


def ranks(cfg, state: GameState) -> dict[int, int]:
    """Rang (0 = premier) de chaque faction."""
    return {faction_id: rank for rank, faction_id in enumerate(ranking(cfg, state))}


def is_over(cfg, state: GameState) -> tuple[bool, bool]:
    """`(terminated, truncated)` au sens PettingZoo.

    `terminated` : il ne reste qu'un survivant (ou aucun) — fin anticipée.
    `truncated`  : le tour maximum est atteint.
    """
    terminated = len(state.living) <= 1
    truncated = state.turn >= int(cfg.world.max_turns)
    return terminated, truncated


def final_report(cfg, state: GameState) -> list[dict]:
    """Résultat de partie, dans l'ordre du classement."""
    order = ranking(cfg, state)
    return [
        {
            "rank": rank,
            "faction_id": faction_id,
            "name": state.faction(faction_id).name,
            "alive": state.faction(faction_id).alive,
            "eliminated_turn": state.faction(faction_id).eliminated_turn,
            "score": round(score(cfg, state, faction_id), 3),
            "components": {
                axis: round(value, 3)
                for axis, value in score_components(state, faction_id).items()
            },
            "reputation": state.faction(faction_id).reputation.to_dict(),
            "traits": {
                key: round(value, 4)
                for key, value in state.faction(faction_id).traits.items()
            },
        }
        for rank, faction_id in enumerate(order)
    ]
