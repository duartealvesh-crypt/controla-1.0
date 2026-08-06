"""La passe de développement : la fonction de transition (spec §8).

C'est **la** pièce maîtresse : elle dicte la vitesse, le développement, et donc tout
le jeu. L'ordre des opérations change l'équilibre du tout au tout — il est figé dans
`rules_text.DEVELOPMENT_PASS_STEPS`, épinglé par un test, et communiqué au dieu dans
son prompt : il doit pouvoir raisonner dessus.

Chaque effet est loggé avec sa **cause**, pour que le dieu voie sa causalité même
médiée (« Bénédiction (rivière nord) : +4.2 nourriture, +3 naissances »).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from controla.diplomacy.commitment import (
    NO_COMMITMENT,
    Pledge,
    enforce_economic_transfers,
    register_pledge,
    verify,
)
from controla.dynamics import economy, emergence, military
from controla.dynamics.drift import apply_drift
from controla.registry.catalog import Catalog
from controla.state.effects import Modifier
from controla.state.game_state import RESOURCE_KEYS, GameState
from controla.rules_text import DEVELOPMENT_PASS_STEPS


@dataclass
class TurnReport:
    """Ce qui s'est passé pendant une passe. C'est la donnée de jeu du replay."""

    turn: int
    events: list[dict] = field(default_factory=list)
    battles: list[dict] = field(default_factory=list)
    eliminations: list[int] = field(default_factory=list)
    per_faction: dict[int, dict] = field(default_factory=dict)
    # Instantané du plateau à la fin de la passe, posé par l'environnement (voir
    # `replay/snapshot.py`). La passe ne le remplit pas elle-même : elle décrit
    # ce qui s'est PASSÉ, l'état qui en résulte est du ressort de l'appelant.
    # Reste `None` quand la restitution n'est pas demandée — ce qui garde les
    # 2 016 parties d'un sweep aussi légères qu'avant.
    board: dict | None = None

    def add(self, *events: dict) -> None:
        self.events.extend(events)

    def extend(self, events: list[dict]) -> None:
        self.events.extend(events)

    def to_dict(self) -> dict:
        payload = {
            "turn": self.turn,
            "events": self.events,
            "battles": self.battles,
            "eliminations": self.eliminations,
            "per_faction": {str(k): v for k, v in self.per_faction.items()},
        }
        # Absent plutôt que `null` : un replay ancien et un replay sans
        # instantané se relisent alors de la même façon.
        if self.board is not None:
            payload["board"] = self.board
        return payload


# ── 1. Effets échus ─────────────────────────────────────────────────────────

def _resolve_effects(cfg, catalog, state, rng, report, signals):
    """Résout les effets datés : messages livrés, engagements formés puis vérifiés.

    §8 range déjà les « tributs » dans cette étape ; les engagements y trouvent donc
    leur place sans ajouter une quinzième étape à un ordre qui est figé.

    Les attaques, elles, sont mises de côté et résolues à l'étape 10, après que la
    production et la Force du tour ont été calculées — le défenseur combat avec la
    force qu'il a *maintenant*.

    La vérification lit `spent_this_turn`, qui n'est purgé qu'à l'étape 12 : elle
    voit donc exactement ce que chaque dieu a financé pendant cette ronde.
    """
    pending_attacks = []
    for effect in state.effects.pop_due(state.turn):
        if effect.kind == "attack":
            pending_attacks.append(effect)
        elif effect.kind == "negotiation":
            pledge = Pledge(
                promiser=effect.actor,
                partner=int(effect.payload["target"]),
                kind=str(effect.payload.get("commitment", NO_COMMITMENT)),
                victim=int(effect.payload.get("victim", -1)),
                turn=state.turn,
                argument=str(effect.payload.get("argument", "")),
            )
            formed = register_pledge(state, pledge, cfg)
            report.add({
                "kind": "negotiation_delivered", "faction": pledge.promiser,
                "target": pledge.partner, "pledge": pledge.kind,
                "argument": pledge.argument, "cause": effect.cause,
            })
            if formed is not None:
                report.add({
                    "kind": "commitment_formed", "commitment": formed.kind,
                    "parties": list(formed.parties),
                    "cause": f"engagement réciproque jusqu'au tour {formed.expires_turn}",
                })
        else:
            report.add({
                "kind": "effect_resolved", "effect": effect.kind,
                "faction": effect.actor, "cause": effect.cause,
            })

    report.extend(verify(cfg, catalog, state))
    enforce_economic_transfers(cfg, state)
    return pending_attacks


# ── 8. Aléa ─────────────────────────────────────────────────────────────────

def _random_events(cfg, state, rng, report, signals):
    """Maladie, catastrophe, découverte.

    L'événement est **annoncé** (visible dans le log) même si son coefficient reste
    caché : un aléa lisible teste le jugement, un aléa opaque teste le dé (§11).
    """
    for faction in state.living:
        stream = rng.stream(state.turn, "events", faction.id)

        if stream.random() < float(cfg.events.disease_probability):
            faction.modifiers["natality"] = Modifier(
                amount=-float(cfg.events.disease_severity),
                until_turn=state.turn + 1,
                cause="épidémie",
            )
            report.add({"kind": "disease", "faction": faction.id,
                        "cause": "épidémie : la natalité chute"})

        if stream.random() < float(cfg.events.disaster_probability):
            severity = float(cfg.events.disaster_severity)
            lost = {}
            for resource in RESOURCE_KEYS:
                amount = faction.stock(resource) * severity
                faction.set_stock(resource, faction.stock(resource) - amount)
                if amount > 0:
                    lost[resource] = round(amount, 2)
            signals[faction.id]["struck"] = True
            # `natural_disaster`, et non `disaster` : ce dernier est le MIRACLE du
            # dieu. Deux choses distinctes ne doivent pas porter le même nom.
            report.add({"kind": "natural_disaster", "faction": faction.id,
                        "lost": lost, "cause": "catastrophe naturelle"})

        if stream.random() < float(cfg.events.discovery_probability):
            amount = float(cfg.events.discovery_amount)
            resource = stream.choice(["food", "wood"])
            faction.set_stock(resource, faction.stock(resource) + amount)
            report.add({"kind": "discovery", "faction": faction.id,
                        "gained": {resource: amount}, "cause": "découverte"})


# ── 10. Guerre ──────────────────────────────────────────────────────────────

def _resolve_attacks(cfg, state, rng, report, signals, attacks):
    """Résout les batailles échues, dans l'ordre déterministe de la file.

    Une alliance de guerre honorée additionne les forces (§7) : si un autre
    attaquant frappe **le même défenseur au même tour**, il est co-belligérant.
    """
    for effect in attacks:
        attacker = state.faction(effect.actor)
        defender = state.faction(effect.payload["defender"])
        if not attacker.alive or not defender.alive:
            continue
        tile = state.grid.get(*effect.payload["tile"])
        if tile.owner != defender.id:
            report.add({"kind": "attack_aborted", "faction": attacker.id,
                        "cause": "la case visée a changé de mains avant l'assaut"})
            continue

        allies = tuple(
            state.faction(other.actor)
            for other in attacks
            if other is not effect
            and other.payload.get("defender") == defender.id
            and other.actor != attacker.id
            and state.faction(other.actor).alive
        )
        stream = rng.stream(state.turn, "battle", attacker.id)
        outcome = military.resolve_battle(
            cfg, state, attacker, defender, tile.coord,
            surprise=bool(effect.payload.get("surprise")),
            stream=stream, allies=allies,
        )
        report.battles.append(outcome.to_dict())
        signals[attacker.id]["fought"] = True
        signals[defender.id]["attacked"] = True
        for ally in allies:
            signals[ally.id]["fought"] = True


# ── 13. Éliminations ────────────────────────────────────────────────────────

def _eliminate(cfg, state, report):
    """Élimination : population épuisée OU plus aucun territoire. Les cases
    redeviennent neutres (§10)."""
    for faction in state.living:
        if faction.population > 0 and state.tile_count(faction.id) > 0:
            continue
        faction.eliminated_turn = state.turn
        for tile in state.tiles_of(faction.id):
            tile.owner = None
            tile.building = None
        report.eliminations.append(faction.id)
        report.add({
            "kind": "elimination", "faction": faction.id,
            "cause": "population épuisée" if faction.population <= 0
                     else "territoire perdu",
        })


# ── La passe ────────────────────────────────────────────────────────────────

def run_development_pass(cfg, catalog: Catalog, state: GameState, rng) -> TurnReport:
    """Applique la passe complète et renvoie ce qui s'est passé.

    L'ordre suit `DEVELOPMENT_PASS_STEPS` à la lettre. Un test compare les étapes
    réellement exécutées à cette liste, pour qu'un réordonnancement soit toujours
    une décision assumée et jamais un effet de bord.
    """
    state.turn += 1
    report = TurnReport(turn=state.turn)
    # Le vécu du tour, collecté au fil de la passe puis consommé par la dérive.
    signals: dict[int, dict] = {faction.id: {} for faction in state.factions}
    executed: list[str] = []

    def step(name: str) -> None:
        executed.append(name)

    # 1 ─ effets échus
    step("resolve_effects")
    attacks = _resolve_effects(cfg, catalog, state, rng, report, signals)

    # 2 ─ production
    step("production")
    production = economy.produce(cfg, catalog, state, rng)
    for faction_id, data in production.items():
        report.per_faction.setdefault(faction_id, {}).update(data)

    # 3 ─ consommation
    step("consumption")
    consumption = economy.consume(cfg, state)
    economy.update_smoothing(cfg, state, production, consumption)
    for faction_id, data in consumption.items():
        report.per_faction.setdefault(faction_id, {}).update(data)

    # 4 ─ population
    step("population")
    growth = economy.grow_population(cfg, state)
    for faction_id, data in growth.items():
        report.per_faction.setdefault(faction_id, {}).update(data)
        signals[faction_id]["grew"] = data["logistic"]

    # 5 ─ recherche
    step("research")
    research = economy.advance_research(cfg, state)
    for faction_id, data in research.items():
        report.per_faction.setdefault(faction_id, {})["research"] = data
        if data["unlocked"]:
            report.add({
                "kind": "research_tier", "faction": faction_id,
                "tier": data["tier"],
                "cause": f"palier de recherche {data['tier']} atteint",
            })

    # 6 ─ force
    step("force")
    military.refresh_force(cfg, state)

    # 7 ─ points d'intervention
    step("intervention")
    economy.recharge_intervention_points(cfg, state)

    # 8 ─ aléa
    step("random_events")
    _random_events(cfg, state, rng, report, signals)

    # 9 ─ comportements émergents
    step("emergence")
    report.extend(emergence.expand(cfg, catalog, state, rng))
    built = emergence.build(cfg, catalog, state, rng)
    report.extend(built)
    for event in built:
        signals[event["faction"]]["built"] = True
    report.extend(emergence.trade(cfg, state, rng))
    report.extend(emergence.trigger_wars(cfg, catalog, state, rng))

    # 10 ─ guerre
    step("war")
    _resolve_attacks(cfg, state, rng, report, signals, attacks)

    # 11 ─ dérive des traits
    step("drift")
    report.extend(apply_drift(cfg, state, rng, signals))

    # 12 ─ décroissance
    step("decay")
    for tile in state.grid.all_tiles():
        tile.expire_modifiers(state.turn)
    for faction in state.factions:
        faction.expire_modifiers(state.turn)
        faction.spent_this_turn.clear()

    # 13 ─ éliminations
    step("elimination")
    _eliminate(cfg, state, report)

    # 14 ─ reward + observations : à la charge de l'environnement
    step("observe")

    expected = [key for key, _label in DEVELOPMENT_PASS_STEPS]
    if executed != expected:
        raise AssertionError(
            f"la passe a dévié de l'ordre déclaré (§8)\n"
            f"  attendu : {expected}\n  exécuté : {executed}"
        )
    return report
