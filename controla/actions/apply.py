"""Exécution des actions du dieu, pilotée par les descripteurs.

Un interpréteur par `EffectSpec.kind`, et **rien d'autre** : ce module ne connaît
aucun nom d'action. Ajouter une action au catalogue la rend jouable sans toucher
ici ; ajouter un *kind* d'effet sans interpréteur fait échouer l'import (voir
l'assertion en bas de fichier), jamais une partie en cours.

**Résolution simultanée** (§1) : `step()` débite les points et enregistre l'action,
mais l'effet n'est appliqué qu'à la barrière de ronde, par `apply_buffered`. Sans
ce tampon, un dieu qui joue tard dans la ronde verrait les coups des autres — la
simultanéité serait un vœu.
"""

from __future__ import annotations

from dataclasses import dataclass

from controla.actions.legality import ActionEntry
from controla.dynamics.emergence import set_trade_relation
from controla.registry.catalog import Catalog
from controla.registry.descriptors import EFFECT_KINDS, EffectSpec
from controla.state.effects import Modifier
from controla.state.game_state import GameState


@dataclass
class SubmittedAction:
    """Une action payée, en attente de la barrière de ronde."""

    faction_id: int
    entry: ActionEntry
    seq: int
    argument: str = ""

    def to_dict(self) -> dict:
        return {
            "faction": self.faction_id,
            "action": self.entry.key,
            "target": self.entry.target,
            "argument": self.argument,
        }


def _duration(cfg, dotted_key: str) -> int:
    node = cfg
    for part in dotted_key.split("."):
        node = getattr(node, part)
    return int(node)


# ── interpréteurs ───────────────────────────────────────────────────────────

def _call_trait(cfg, faction, trait: str, amount: float) -> float:
    """Déplace un trait d'une fraction de la MARGE RESTANTE jusqu'à la borne.

    Saturation automatique dans les deux sens (§4) : plus un trait est haut, moins
    un Appel le fait monter. Cela empêche le runaway ET rend le spam d'une seule
    action inefficace, donc force la diversification.
    """
    low, high = float(cfg.traits.min), float(cfg.traits.max)
    before = faction.traits[trait]
    if amount > 0:
        delta = amount * float(cfg.traits.call_step) * (high - before)
    else:
        delta = amount * float(cfg.traits.call_penalty) * (before - low)
    faction.traits[trait] = max(low, min(high, before + delta))
    return faction.traits[trait] - before


def _effect_trait_call(cfg, catalog, state, action, effect) -> dict:
    faction = state.faction(action.faction_id)
    moved = _call_trait(cfg, faction, effect.trait, effect.amount)
    return {"trait": effect.trait, "moved": round(moved, 4), "on": faction.id}


def _effect_trait_call_foe(cfg, catalog, state, action, effect) -> dict:
    tile = state.grid.get(*action.entry.target)
    if tile.owner is None:
        return {"skipped": "case sans propriétaire"}
    faction = state.faction(tile.owner)
    moved = _call_trait(cfg, faction, effect.trait, effect.amount)
    return {"trait": effect.trait, "moved": round(moved, 4), "on": faction.id}


def _effect_war_desire(cfg, catalog, state, action, effect) -> dict:
    """Pousse l'envie de guerre. La modulation par le trait Guerrier a lieu au
    déclenchement (`emergence.trigger_wars`), pas ici : la compter deux fois
    ferait de l'Appel à la guerre une arme au rendement croissant."""
    faction = state.faction(action.faction_id)
    target = int(action.entry.target)
    boost = effect.amount * float(cfg.emergence.war_call_boost)
    faction.war_desire[target] = faction.war_desire.get(target, 0.0) + boost
    return {"target": target, "war_desire": round(faction.war_desire[target], 3)}


def _effect_trade_relation(cfg, catalog, state, action, effect) -> dict:
    """La relation commerciale est symétrique : un embargo d'un côté coupe les
    échanges des deux côtés (§6)."""
    faction = state.faction(action.faction_id)
    target = int(action.entry.target)
    before = faction.trade_relation.get(target, 0.0)
    set_trade_relation(state, faction.id, target, before + effect.amount)
    return {"target": target, "trade_relation": round(faction.trade_relation[target], 3)}


def _effect_tile_modifier(cfg, catalog, state, action, effect) -> dict:
    """Pose un modificateur daté. La clé inclut l'acteur, pour que deux dieux
    frappant la même case cumulent au lieu de s'écraser."""
    tile = state.grid.get(*action.entry.target)
    key = f"{effect.modifier}:{action.faction_id}"
    tile.modifiers[key] = Modifier(
        amount=effect.amount,
        until_turn=state.turn + _duration(cfg, effect.duration_key),
        cause=f"{action.entry.key} de {state.faction(action.faction_id).name}",
    )
    return {"tile": list(tile.coord), "modifier": key, "amount": effect.amount}


def _effect_reveal(cfg, catalog, state, action, effect) -> dict:
    faction = state.faction(action.faction_id)
    faction.revealed_until_turn = state.turn + _duration(cfg, effect.duration_key)
    return {"revealed_until_turn": faction.revealed_until_turn}


def _effect_open_channel(cfg, catalog, state, action, effect) -> dict:
    """Ouvre un canal de négociation : le message arrive au tour suivant (§7).

    L'argumentaire et le commitment voyagent dans la file d'effets datés, comme
    tout le reste — pas de canal parallèle à maintenir.
    """
    target, commitment, victim = action.entry.target
    state.effects.add(
        "negotiation",
        due_turn=state.turn + action.entry.spec.latency,
        actor=action.faction_id,
        cause=f"négociation ouverte avec {state.faction(target).name}",
        target=target,
        argument=action.argument,
        commitment=commitment,
        victim=victim,
    )
    return {
        "target": target, "commitment": commitment, "victim": victim,
        "argument_length": len(action.argument),
    }


_INTERPRETERS = {
    "trait_call": _effect_trait_call,
    "trait_call_foe": _effect_trait_call_foe,
    "war_desire": _effect_war_desire,
    "trade_relation": _effect_trade_relation,
    "tile_modifier": _effect_tile_modifier,
    "reveal": _effect_reveal,
    "open_channel": _effect_open_channel,
}

# Le contrat descripteur <-> moteur, vérifié à l'import : un kind déclaré sans
# interpréteur (ou l'inverse) fait échouer le démarrage, pas une partie.
assert set(_INTERPRETERS) == set(EFFECT_KINDS), (
    "désaccord entre EFFECT_KINDS et les interpréteurs : "
    f"déclarés sans interpréteur {sorted(EFFECT_KINDS - set(_INTERPRETERS))}, "
    f"interpréteurs non déclarés {sorted(set(_INTERPRETERS) - EFFECT_KINDS)}"
)


# ── soumission et barrière ──────────────────────────────────────────────────

def submit(
    cfg, state: GameState, faction_id: int, entry: ActionEntry, seq: int,
    argument: str = "",
) -> SubmittedAction:
    """Débite les points et enregistre l'action, SANS appliquer son effet.

    Le registre `spent_this_turn` est la preuve d'intention qui sert à vérifier les
    commitments (§7 corrigé) : c'est l'action payée qui engage le dieu, pas son issue.
    """
    faction = state.faction(faction_id)
    faction.intervention_points -= entry.spec.cost
    faction.spent_this_turn.append((entry.key, entry.target))
    return SubmittedAction(
        faction_id=faction_id, entry=entry, seq=seq, argument=argument,
    )


def apply_buffered(
    cfg, catalog: Catalog, state: GameState, buffer: list[SubmittedAction]
) -> list[dict]:
    """Applique toutes les actions de la ronde, à la barrière.

    Ordre déterministe `(faction, seq)` : le replay est fidèle, et aucun dieu n'est
    avantagé par sa position dans la ronde.
    """
    events: list[dict] = []
    for action in sorted(buffer, key=lambda item: (item.faction_id, item.seq)):
        # Un événement par ACTION, portant tous ses effets — et non un par effet :
        # une bénédiction est un seul geste du dieu, même si elle touche à la fois
        # la case et le trait Travailleur.
        effects = [
            {
                "effect": effect.kind,
                "detail": _INTERPRETERS[effect.kind](cfg, catalog, state, action, effect),
            }
            for effect in action.entry.spec.effects
        ]
        events.append({
            "kind": "divine_action",
            "faction": action.faction_id,
            "action": action.entry.key,
            "target": action.entry.target,
            "effects": effects,
            "cause": f"{action.entry.key} de {state.faction(action.faction_id).name}",
        })
    return events


def describe_effect(cfg, catalog: Catalog, effect: EffectSpec) -> str:
    """Passerelle pour le prompt : la description vient du descripteur lui-même."""
    return effect.describe(cfg, catalog)
