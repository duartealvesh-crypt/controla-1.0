"""Observation, brouillard, et espaces gymnasium (spec §11).

Deux publics, une seule vérité :

* `observe()` renvoie une observation **numérique de forme fixe** — c'est ce dont
  une policy RL a besoin, et c'est ce que `pettingzoo.test.api_test` valide.
* `build_view()` renvoie la **même information** sous forme structurée lisible,
  transportée dans `infos`. La policy LLM en fait du texte (tranche 2).

Le partage public/caché suit §11 :

| Public                                   | Caché                                  |
|------------------------------------------|----------------------------------------|
| carte, terrains, propriété des cases      | traits, stocks, Force réelle, Recherche |
| bâtiments visibles                        | armées en transit                      |
| compteur de trahisons de chaque dieu      | score exact                            |
| guerres déclarées                         | contenu des négociations privées       |

Le classement n'est donc qu'*estimable* (territoire, bâtiments visibles) : la
coalition anti-leader reste possible sans tuer le bluff. L'Omniscience lève le
brouillard pour un tour.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from controla.actions.legality import ActionIndex, legal_entries, legal_mask
from controla.dynamics import military
from controla.registry.catalog import Catalog
from controla.state.game_state import GameState

# Ce qu'un dieu sait de lui-même, dans l'ordre du vecteur `self`.
SELF_FEATURES = (
    "food", "wood", "metal", "population", "research", "research_tier",
    "force", "hunger", "intervention_points", "tiles", "buildings",
    "incoming_attacks", "pending_effects",
)
# Ce que tout le monde voit de chacun.
PUBLIC_FEATURES = (
    "alive", "tiles", "buildings", "betrayed", "honored",
    "at_war_with_me", "trade_relation_with_me", "is_me",
)
# Ce que seule l'Omniscience révèle (0 partout sinon).
REVEALED_FEATURES = ("population", "force", "research", "food", "wood", "metal")

_BIG = 1e6


def observation_space(cfg, catalog: Catalog, index: ActionIndex) -> spaces.Dict:
    size = int(cfg.world.grid_size)
    count = int(cfg.world.num_factions)
    traits = len(catalog.trait_keys)
    return spaces.Dict({
        "action_mask": spaces.MultiBinary(len(index)),
        "self": spaces.Box(-_BIG, _BIG, shape=(len(SELF_FEATURES),), dtype=np.float32),
        "traits": spaces.Box(
            float(cfg.traits.min), float(cfg.traits.max), shape=(traits,), dtype=np.float32
        ),
        "public": spaces.Box(-_BIG, _BIG, shape=(count, len(PUBLIC_FEATURES)), dtype=np.float32),
        "revealed": spaces.Box(-_BIG, _BIG, shape=(count, len(REVEALED_FEATURES)), dtype=np.float32),
        "revealed_mask": spaces.MultiBinary(count),
        "terrain": spaces.Box(0, len(catalog.terrains), shape=(size, size), dtype=np.int8),
        "owner": spaces.Box(-1, count, shape=(size, size), dtype=np.int8),
        "building": spaces.Box(-1, len(catalog.buildings), shape=(size, size), dtype=np.int8),
        "turn": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
    })


def action_space(cfg, index: ActionIndex) -> spaces.Discrete:
    """Un `Discrete` plat, masqué par `observation["action_mask"]`.

    C'est le contrat RL canonique : toute bibliothèque sait échantillonner
    `action_space.sample(mask)`. Un espace `Dict` casserait cette convention, que
    `pettingzoo.test.api_test` vérifie d'ailleurs explicitement.

    L'ENGAGEMENT diplomatique est replié dans l'index (une entrée `negotiate` par
    couple partenaire × type d'engagement), ce qui le rend masquable au passage.

    L'ARGUMENTAIRE en texte libre ne peut pas vivre dans un `Discrete`. `step()`
    accepte donc aussi un dict `{"action": int, "argument": str}` — un sur-ensemble
    documenté du contrat, utilisé par les policies textuelles. Une policy RL n'a
    jamais à le connaître.
    """
    return spaces.Discrete(len(index))


# ── brouillard ──────────────────────────────────────────────────────────────

def sees(state: GameState, viewer_id: int, target_id: int) -> bool:
    """Le spectateur voit-il l'intérieur de la faction cible ?

    Soi-même toujours ; les autres seulement sous Omniscience. C'est le fog
    « option 1 » de §11 : géographiquement tout est visible, mais les intérieurs
    (stocks, traits, Force) sont opaques.
    """
    if viewer_id == target_id:
        return True
    return state.faction(viewer_id).revealed_until_turn >= state.turn


# ── observation numérique ───────────────────────────────────────────────────

def observe(cfg, catalog: Catalog, state: GameState, index: ActionIndex,
            faction_id: int) -> dict:
    size = int(cfg.world.grid_size)
    count = int(cfg.world.num_factions)
    me = state.faction(faction_id)
    terrain_keys = catalog.terrain_keys
    building_keys = catalog.building_keys

    terrain = np.zeros((size, size), dtype=np.int8)
    owner = np.full((size, size), -1, dtype=np.int8)
    building = np.full((size, size), -1, dtype=np.int8)
    for tile in state.grid.all_tiles():
        terrain[tile.x, tile.y] = terrain_keys.index(tile.terrain)
        if tile.owner is not None:
            owner[tile.x, tile.y] = tile.owner
        if tile.building is not None:
            building[tile.x, tile.y] = building_keys.index(tile.building)

    incoming = len([
        effect for effect in state.effects.pending(kind="attack")
        if effect.payload.get("defender") == faction_id
        and (me.revealed_until_turn >= state.turn)
    ])
    own = np.array([
        me.food, me.wood, me.metal, me.population, me.research, me.research_tier,
        me.force, me.hunger, me.intervention_points,
        state.tile_count(faction_id), state.building_total(faction_id),
        incoming, len(state.effects.pending(actor=faction_id)),
    ], dtype=np.float32)

    public = np.zeros((count, len(PUBLIC_FEATURES)), dtype=np.float32)
    revealed = np.zeros((count, len(REVEALED_FEATURES)), dtype=np.float32)
    revealed_mask = np.zeros(count, dtype=np.int8)
    for other in state.factions:
        public[other.id] = [
            1.0 if other.alive else 0.0,
            state.tile_count(other.id),
            state.building_total(other.id),
            other.reputation.betrayed,
            other.reputation.honored,
            1.0 if faction_id in other.at_war_with else 0.0,
            me.trade_relation.get(other.id, 0.0),
            1.0 if other.id == faction_id else 0.0,
        ]
        if sees(state, faction_id, other.id):
            revealed_mask[other.id] = 1
            revealed[other.id] = [
                other.population, other.force, other.research,
                other.food, other.wood, other.metal,
            ]

    return {
        "action_mask": np.array(
            legal_mask(cfg, state, faction_id, index), dtype=np.int8
        ),
        "self": own,
        "traits": np.array(
            [me.traits[key] for key in catalog.trait_keys], dtype=np.float32
        ),
        "public": public,
        "revealed": revealed,
        "revealed_mask": revealed_mask,
        "terrain": terrain,
        "owner": owner,
        "building": building,
        "turn": np.array(
            [state.turn / max(1, int(cfg.world.max_turns))], dtype=np.float32
        ),
    }


# ── vue structurée (pour le prompt LLM et le replay) ────────────────────────

def _tile_facts(state: GameState, target) -> dict | None:
    """Ce qu'un dieu voit d'une case visée : tout cela est public (§11).

    Sans cela, un miracle se visait à l'aveugle : le prompt ne donnait que des
    coordonnées, et l'on a mesuré des désastres tombant sur le mauvais rival.
    """
    if not (isinstance(target, tuple) and len(target) == 2):
        return None
    tile = state.grid.get(*target)
    return {
        "owner": None if tile.owner is None else state.faction(tile.owner).name,
        "owner_id": tile.owner,
        "terrain": tile.terrain,
        "building": tile.building,
    }


def turn_digest(catalog: Catalog, state: GameState, report, faction_id: int) -> dict | None:
    """Ce que CE dieu a pu constater du dernier tour, en lignes lisibles.

    Sans ce compte rendu, un dieu ne voyait que l'état présent : un modèle a écrit
    « sans incident » pendant qu'on lui prenait deux cases, ses expansions masquant
    ses pertes.

    Le brouillard reste entier : ses propres actes, ce qui lui est arrivé, et ce
    qui est public. Ni la Force ni les pertes adverses, ni les coups des rivaux,
    ni les armées en transit. Un désastre subi est constaté sans son auteur.
    """
    if report is None:
        return None

    def name(other: int) -> str:
        return state.faction(other).name

    def place(coord) -> str:
        return f"({coord[0]},{coord[1]})"

    def target_text(target) -> str:
        if target is None:
            return ""
        if isinstance(target, int):
            return f" sur {name(target)}"
        if len(target) == 2:
            return f" en {place(target)}"
        partner, commitment, _victim = target
        return f" vers {name(partner)} ({commitment})"

    played: list[str] = []
    lines: list[str] = []
    for event in report.events:
        kind = event.get("kind")
        mine = event.get("faction") == faction_id
        if kind == "divine_action":
            target = event.get("target")
            label = catalog.label(event["action"])
            if mine:
                played.append(label + target_text(target))
            elif (
                isinstance(target, (tuple, list)) and len(target) == 2
                and state.grid.get(*target).owner == faction_id
            ):
                lines.append(f"Ta case {place(target)} a subi : {label}.")
        elif kind == "expansion" and mine:
            lines.append(f"Ton peuple a colonisé {place(event['tile'])}.")
        elif kind == "build" and mine:
            lines.append(
                f"Ton peuple a bâti {catalog.label(event['building'])} "
                f"en {place(event['tile'])}."
            )
        elif kind == "research_tier" and mine:
            lines.append(f"Palier de recherche {event['tier']} atteint.")
        elif kind == "trade" and faction_id in event["factions"]:
            partner = next(f for f in event["factions"] if f != faction_id)
            lines.append(f"Échange commercial avec {name(partner)}.")
        elif kind == "war_launched" and mine:
            lines.append(
                f"Ton peuple a lancé un assaut sur {name(event['target'])} "
                f"en {place(event['tile'])}."
            )
        elif kind in ("disease", "natural_disaster", "discovery") and mine:
            lines.append(f"Événement : {event['cause']}.")
        elif kind == "commitment_formed" and faction_id in event["parties"]:
            partner = next(f for f in event["parties"] if f != faction_id)
            lines.append(f"Engagement {event['commitment']} conclu avec {name(partner)}.")
        elif kind == "commitment_betrayed" and faction_id in event.get("parties", ()):
            who = "Tu as" if mine else f"{name(event['faction'])} a"
            lines.append(f"{who} rompu l'engagement {event['commitment']}.")
        elif kind == "elimination":
            lines.append(f"{name(event['faction'])} est éliminé.")

    for battle in report.battles:
        allied = faction_id in battle.get("allies", ())
        if faction_id == battle["attacker"] or allied:
            enemy = name(battle["defender"])
            won = battle["attacker_won"]
            text = f"Bataille contre {enemy} (tu attaquais) : {'victoire' if won else 'défaite'}"
            if faction_id == battle["attacker"]:
                text += f", {battle['attacker_losses']:.1f} morts chez toi"
        elif faction_id == battle["defender"]:
            enemy = name(battle["attacker"])
            won = not battle["attacker_won"]
            text = (
                f"{enemy} t'a attaqué : {'victoire' if won else 'défaite'}, "
                f"{battle['defender_losses']:.1f} morts chez toi"
            )
        else:
            continue
        if battle.get("annexed"):
            text += f", case {place(battle['annexed'])} annexée par {name(battle['attacker'])}"
        lines.append(text + ".")

    return {"turn": report.turn, "played": played, "lines": lines}


def build_view(cfg, catalog: Catalog, state: GameState, index: ActionIndex,
               faction_id: int, last_report=None) -> dict:
    """La même information, sous une forme lisible. Transportée par `infos`."""
    me = state.faction(faction_id)
    entries = legal_entries(cfg, state, faction_id, index)

    rivals = []
    for other in state.factions:
        if other.id == faction_id:
            continue
        block = {
            "id": other.id,
            "name": other.name,
            "alive": other.alive,
            "eliminated_turn": other.eliminated_turn,
            "tiles": state.tile_count(other.id),
            "buildings": state.building_counts(other.id),
            "at_war_with_me": faction_id in other.at_war_with,
            "trade_relation": round(me.trade_relation.get(other.id, 0.0), 3),
            "reputation": other.reputation.to_dict(),
        }
        if sees(state, faction_id, other.id):
            block["revealed"] = {
                "population": round(other.population, 1),
                "force": round(other.force, 1),
                "research": round(other.research, 1),
                "research_tier": other.research_tier,
                "food": round(other.food, 1),
                "wood": round(other.wood, 1),
                "metal": round(other.metal, 1),
                "traits": {k: round(v, 3) for k, v in other.traits.items()},
            }
        rivals.append(block)

    return {
        "turn": state.turn,
        "max_turns": int(cfg.world.max_turns),
        "faction_id": faction_id,
        "self": {
            "name": me.name,
            "population": round(me.population, 1),
            "food": round(me.food, 1),
            "wood": round(me.wood, 1),
            "metal": round(me.metal, 1),
            "research": round(me.research, 1),
            "research_tier": me.research_tier,
            "force": round(me.force, 1),
            "hunger": round(me.hunger, 3),
            "engaged_fraction": round(military.engaged_fraction(cfg, me), 3),
            "intervention_points": me.intervention_points,
            "tiles": state.tile_count(faction_id),
            "buildings": state.building_counts(faction_id),
            "traits": {k: round(v, 3) for k, v in me.traits.items()},
            "war_desire": {str(k): round(v, 3) for k, v in me.war_desire.items()},
            "reputation": me.reputation.to_dict(),
        },
        "rivals": rivals,
        # Sa propre file est visible ; celle des autres est cachée. Espionner,
        # c'est lire la file adverse — d'où l'intérêt de l'Omniscience.
        "pending": [effect.to_dict() for effect in state.effects.pending(actor=faction_id)],
        # Les messages REÇUS, avec leur argumentaire. Le contenu d'une négociation
        # reste privé aux deux parties (§11) — c'est ce qui rend le bluff possible.
        "inbox": [
            {
                "from": pledge.promiser,
                "from_name": state.faction(pledge.promiser).name,
                "pledge": pledge.kind,
                "victim": pledge.victim,
                "victim_name": (
                    state.faction(pledge.victim).name if pledge.victim >= 0 else None
                ),
                "argument": pledge.argument,
                "turn": pledge.turn,
            }
            for pledge in state.negotiations
            if pledge.partner == faction_id
            and pledge.turn >= state.turn - 1        # ce qui vient d'arriver
        ],
        "commitments": [
            {
                **commitment.to_dict(),
                "partner": commitment.other(faction_id),
                "partner_name": state.faction(commitment.other(faction_id)).name,
            }
            for commitment in state.commitments
            if faction_id in commitment.parties
        ],
        "legal_actions": [
            {
                "index": entry.index,
                "action": entry.key,
                "target": entry.target,
                "cost": entry.spec.cost,
                "label": entry.describe(state),
                "tile": _tile_facts(state, entry.target),
            }
            for entry in entries
        ],
        "last_turn": turn_digest(catalog, state, last_report, faction_id),
    }
