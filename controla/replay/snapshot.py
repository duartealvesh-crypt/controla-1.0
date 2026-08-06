"""Instantanés de partie destinés à la restitution graphique (§14.6).

Pourquoi enregistrer l'état plutôt que le reconstruire
-----------------------------------------------------
Une carte de replay peut se déduire des événements : `expansion` donne une case
et son nouveau propriétaire, `build` un bâtiment, une bataille son annexion. La
tentation est forte, puisqu'aucune donnée nouvelle n'est à écrire.

C'est un piège. La déduction n'est exacte que tant qu'AUCUNE mécanique ne
déplace une case sans émettre l'événement correspondant — l'élimination rend son
territoire neutre, une prise détruit ou capture un ouvrage selon un tirage. Le
jour où une règle nouvelle oublie son événement, la carte du replay diverge de
la partie **en silence**, et l'on croit lire une partie qu'on n'a pas jouée.
C'est exactement le mode de panne que ce projet refuse ailleurs (l'aléa keyé, le
no-op compté plutôt que le repli aléatoire).

L'instantané est donc **autoritatif** : ce qui est affiché est ce qui était.

Le coût est négligeable et mesurable : deux entiers par case, soit 2 × 121
nombres par tour, ~10 ko sur une partie de quinze tours dont le JSONL en pèse
déjà 125.

Ce que porte quoi
-----------------
* `world_snapshot` — l'**immuable** : taille, terrains, tables de clés. Écrit
  une fois par partie.
* `board_snapshot` — le **mutable** : propriété, ouvrages, et l'état chiffré de
  chaque faction. Écrit une fois par tour. C'est lui qui rend traçables les
  courbes de population, de score et de recherche, que les rapports de tour ne
  donnent qu'en variations.
"""

from __future__ import annotations

from controla.dynamics.scoring import score, score_components
from controla.registry.catalog import Catalog
from controla.replay.render import building_glyph
from controla.state.game_state import GameState

# Case sans propriétaire / sans ouvrage. `null` serait plus explicite en JSON,
# mais un entier garde les deux grilles homogènes et compactes.
NONE_INDEX = -1


def world_snapshot(cfg, catalog: Catalog, state: GameState) -> dict:
    """La part immuable d'une partie : la carte et les tables de clés.

    Les tables (`terrain_keys`, `building_keys`) accompagnent les grilles pour
    que la restitution reste lisible sans importer le catalogue — un replay
    stocké doit pouvoir se relire seul, y compris par un navigateur.
    """
    size = state.grid.size
    terrain_keys = list(catalog.terrain_keys)
    return {
        "size": size,
        "max_turns": int(cfg.world.max_turns),
        "terrain_keys": terrain_keys,
        "building_keys": list(catalog.building_keys),
        "trait_keys": list(catalog.trait_keys),
        # Rangée par rangée (y extérieur, x intérieur), l'ordre naturel d'un
        # affichage : la ligne 0 du tableau est la ligne 0 de l'écran.
        "terrain": [
            terrain_keys.index(state.grid.get(x, y).terrain)
            for y in range(size)
            for x in range(size)
        ],
        "factions": [
            {"id": faction.id, "name": faction.name} for faction in state.factions
        ],
        # Les mêmes glyphes que le rendu texte, et pas une seconde table : deux
        # bâtiments dont le libellé commence pareil (Camp militaire / Champs)
        # deviendraient sinon indiscernables sur la carte.
        "glyphs": {key: building_glyph(key) for key in catalog.building_keys},
        "labels": {
            key: catalog.label(key)
            for key in (
                *catalog.terrain_keys, *catalog.building_keys,
                *catalog.trait_keys, *(spec.key for spec in catalog.actions),
                *(spec.key for spec in catalog.resources),
            )
        },
    }


def board_snapshot(cfg, catalog: Catalog, state: GameState) -> dict:
    """L'état du plateau et des factions à la fin d'un tour."""
    size = state.grid.size
    building_keys = list(catalog.building_keys)

    owner: list[int] = []
    building: list[int] = []
    for y in range(size):
        for x in range(size):
            tile = state.grid.get(x, y)
            owner.append(NONE_INDEX if tile.owner is None else tile.owner)
            building.append(
                NONE_INDEX if tile.building is None
                else building_keys.index(tile.building)
            )

    return {
        "owner": owner,
        "building": building,
        "factions": [_faction_snapshot(cfg, state, faction.id) for faction in state.factions],
    }


def _faction_snapshot(cfg, state: GameState, faction_id: int) -> dict:
    """Le chiffré d'une faction — vue d'arbitre, sans brouillard.

    C'est un replay : le brouillard sert à jouer, pas à relire. Cacher ici ce
    qu'un dieu ignorait rendrait le replay illisible sans rien protéger, la
    partie étant finie.
    """
    faction = state.faction(faction_id)
    return {
        "id": faction.id,
        "alive": faction.alive,
        "eliminated_turn": faction.eliminated_turn,
        "population": round(faction.population, 2),
        "food": round(faction.food, 2),
        "wood": round(faction.wood, 2),
        "metal": round(faction.metal, 2),
        "research": round(faction.research, 2),
        "research_tier": faction.research_tier,
        "force": round(faction.force, 2),
        "hunger": round(faction.hunger, 3),
        "intervention_points": faction.intervention_points,
        "tiles": state.tile_count(faction_id),
        "buildings": state.building_counts(faction_id),
        "traits": {key: round(value, 4) for key, value in faction.traits.items()},
        "trade_relation": {str(k): round(v, 3) for k, v in faction.trade_relation.items()},
        "war_desire": {str(k): round(v, 3) for k, v in faction.war_desire.items()},
        "at_war_with": sorted(faction.at_war_with),
        "reputation": faction.reputation.to_dict(),
        "score": round(score(cfg, state, faction_id), 2),
        "components": {
            axis: round(value, 2)
            for axis, value in score_components(state, faction_id).items()
        },
    }
