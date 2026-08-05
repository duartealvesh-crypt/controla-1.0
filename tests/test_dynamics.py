"""Les dynamiques du moteur : Force, combat, envies, dérive, économie.

Ces tests visent en priorité les endroits où la spec a dû être **corrigée** ou
levée d'ambiguïté — c'est là que le risque de régression est le plus élevé.
"""

from __future__ import annotations

import math

import pytest

from controla.dynamics import economy, emergence, military
from controla.dynamics.drift import apply_drift
from controla.registry.catalog import build_catalog
from controla.rng import KeyedRNG
from controla.state.game_state import create_state
from tests.helpers import base_config


@pytest.fixture
def world():
    cfg = base_config()
    catalog = build_catalog(cfg)
    rng = KeyedRNG(1)
    return cfg, catalog, create_state(cfg, catalog, rng), rng


# ── Force (§3, corrigée : apports technologiques saturants) ─────────────────

def test_force_is_never_negative_even_at_minimum_traits(world):
    cfg, _catalog, state, _rng = world
    faction = state.faction(0)
    faction.traits["warrior"] = float(cfg.traits.min)
    faction.hunger = 1.0
    assert military.compute_force(cfg, state, faction) >= 0.0


def test_hunger_is_a_multiplicative_malus(world):
    """La faim ne peut pas rendre la Force négative — elle la met à zéro (§3)."""
    cfg, _catalog, state, _rng = world
    faction = state.faction(0)
    faction.hunger = 0.0
    full = military.compute_force(cfg, state, faction)
    faction.hunger = 0.5
    assert military.compute_force(cfg, state, faction) == pytest.approx(full * 0.5)
    faction.hunger = 1.0
    assert military.compute_force(cfg, state, faction) == 0.0


def test_metal_contribution_saturates_instead_of_exploding(world):
    """LA correction du plan : `β·métal` deviendrait quadratique en territoire.

    Avec un terme saturant, doubler le métal n'ajoute jamais autant que la première
    moitié, et l'apport reste borné par `metal_weight` — pas de scaling exponentiel,
    donc pas de stratégie dominante créée par la mécanique (§12).
    """
    cfg, _catalog, state, _rng = world
    faction = state.faction(0)
    faction.hunger = 0.0

    def force_with(metal: float) -> float:
        faction.metal = metal
        return military.compute_force(cfg, state, faction)

    base = force_with(0.0)
    gains = [force_with(m) - base for m in (40.0, 400.0, 4000.0)]
    assert gains[0] < gains[1] < gains[2], "l'apport métal devrait croître"

    ceiling = base + faction.population * float(cfg.force.metal_weight)
    assert force_with(10 ** 9) < ceiling * 1.001, "l'apport métal n'est pas plafonné"
    # Rendements décroissants : le second saut rapporte moins que le premier.
    assert (gains[1] - gains[0]) < gains[0]


def test_camp_contribution_also_saturates(world):
    cfg, _catalog, state, _rng = world
    faction = state.faction(0)
    faction.hunger = 0.0
    tiles = state.tiles_of(faction.id)

    base = military.compute_force(cfg, state, faction)
    tiles[0].building = "camp"
    one = military.compute_force(cfg, state, faction)
    tiles[1].building = "camp"
    two = military.compute_force(cfg, state, faction)
    tiles[2].building = "camp"
    three = military.compute_force(cfg, state, faction)

    assert base < one < two < three
    assert (three - two) < (two - one) < (one - base)


def test_engaged_fraction_follows_the_warrior_trait(world):
    """Le dieu influence l'ardeur, jamais le nombre (§9)."""
    cfg, _catalog, state, _rng = world
    faction = state.faction(0)

    faction.traits["warrior"] = float(cfg.traits.min)
    assert military.engaged_fraction(cfg, faction) == pytest.approx(cfg.force.engaged_min)
    faction.traits["warrior"] = float(cfg.traits.max)
    assert military.engaged_fraction(cfg, faction) == pytest.approx(cfg.force.engaged_max)
    faction.traits["warrior"] = 0.0
    middle = military.engaged_fraction(cfg, faction)
    assert cfg.force.engaged_min < middle < cfg.force.engaged_max


# ── Combat de Tullock (§9) ──────────────────────────────────────────────────

def test_win_probability_is_scale_invariant(world):
    """Seul le RATIO compte : c'est la propriété qui rend le combat lisible."""
    cfg = world[0]
    for scale in (0.01, 1.0, 1000.0):
        assert military.win_probability(cfg, 200 * scale, 100 * scale, False) == \
               pytest.approx(military.win_probability(cfg, 200, 100, False))


def test_win_probability_is_a_half_at_parity(world):
    cfg = world[0]
    assert military.win_probability(cfg, 50, 50, False) == pytest.approx(0.5)


def test_decisiveness_is_the_chance_skill_dial(world):
    """`m` est LE curseur : à m=1 une armée double gagne 67 %, à m élevé bien plus."""
    lottery = base_config(combat={"decisiveness": 1.0})
    skill = base_config(combat={"decisiveness": 4.0})
    assert military.win_probability(lottery, 200, 100, False) == pytest.approx(2 / 3)
    assert military.win_probability(skill, 200, 100, False) > 0.93
    assert military.win_probability(world[0], 200, 100, False) > \
           military.win_probability(lottery, 200, 100, False)


def test_surprise_only_helps_when_the_attack_was_undetected(world):
    cfg = world[0]
    seen = military.win_probability(cfg, 100, 100, surprise=False)
    unseen = military.win_probability(cfg, 100, 100, surprise=True)
    assert unseen > seen, "la surprise devrait récompenser une traîtrise réussie"
    assert seen == pytest.approx(0.5)


def test_win_probability_handles_the_degenerate_cases(world):
    cfg = world[0]
    assert military.win_probability(cfg, 0.0, 10.0, False) == 0.0
    assert military.win_probability(cfg, 10.0, 0.0, False) == 1.0


def test_the_winner_bleeds_too(world):
    """Frein anti-snowball : gagner coûte, donc on ne roule pas sur la carte (§9)."""
    cfg, _catalog, state, rng = world
    attacker, defender = state.faction(0), state.faction(1)
    attacker.force, defender.force = 400.0, 60.0
    before = attacker.population
    target = next(tile.coord for tile in state.tiles_of(defender.id))

    outcome = military.resolve_battle(
        cfg, state, attacker, defender, target,
        surprise=False, stream=rng.stream(1, "battle", 0),
    )
    assert outcome.attacker_won
    assert attacker.population < before, "le vainqueur n'a subi aucune perte"
    assert outcome.defender_losses > outcome.attacker_losses


def test_a_narrow_victory_costs_more_than_a_crushing_one(world):
    """L'attrition dépend du ratio de forces (§9)."""
    cfg = base_config(combat={"attrition_sigma": 0.0})
    catalog = build_catalog(cfg)

    def losses(attack: float, defend: float) -> float:
        state = create_state(cfg, catalog, KeyedRNG(2))
        attacker, defender = state.faction(0), state.faction(1)
        attacker.force, defender.force = attack, defend
        attacker.population = 100.0
        target = next(tile.coord for tile in state.tiles_of(defender.id))
        outcome = military.resolve_battle(
            cfg, state, attacker, defender, target,
            surprise=False, stream=KeyedRNG(2).stream(1, "battle", 0),
        )
        return outcome.attacker_losses if outcome.attacker_won else float("nan")

    crushing = losses(10_000.0, 1.0)
    narrow = losses(105.0, 100.0)
    assert not math.isnan(crushing) and not math.isnan(narrow)
    assert narrow > crushing, "une victoire de justesse devrait coûter plus cher"


def test_a_won_battle_annexes_the_targeted_tile(world):
    cfg, _catalog, state, rng = world
    attacker, defender = state.faction(0), state.faction(1)
    attacker.force, defender.force = 5000.0, 1.0
    target = next(tile.coord for tile in state.tiles_of(defender.id))

    outcome = military.resolve_battle(
        cfg, state, attacker, defender, target,
        surprise=False, stream=rng.stream(1, "battle", 0),
    )
    assert outcome.attacker_won and outcome.annexed == target
    assert state.grid.get(*target).owner == attacker.id
    assert state.grid.get(*target).building is None


def test_allies_forces_add_up_and_share_the_loot(world):
    """Alliance de guerre honorée : forces additionnées, butin partagé (§7)."""
    cfg, _catalog, state, rng = world
    attacker, ally, defender = state.faction(0), state.faction(2), state.faction(1)
    attacker.force, ally.force, defender.force = 100.0, 100.0, 50.0
    defender.food = 200.0
    ally_food_before = ally.food
    target = next(tile.coord for tile in state.tiles_of(defender.id))

    outcome = military.resolve_battle(
        cfg, state, attacker, defender, target,
        surprise=False, stream=rng.stream(1, "battle", 0), allies=(ally,),
    )
    assert outcome.allies == (ally.id,)
    assert outcome.attacker_force == pytest.approx(
        military.engaged_force(cfg, attacker) + military.engaged_force(cfg, ally)
    )
    if outcome.attacker_won:
        assert ally.food > ally_food_before, "l'allié n'a pas reçu sa part du butin"


# ── Envies (§5) ─────────────────────────────────────────────────────────────

def test_deficit_uses_smoothed_averages(world):
    """Le lissage évite qu'une mauvaise récolte fasse spiker l'envie (§5)."""
    _cfg, _catalog, state, _rng = world
    faction = state.faction(0)
    faction.ema_consumption["food"] = 10.0
    faction.ema_production["food"] = 10.0
    assert emergence.relative_deficit(faction, "food") == 0.0
    faction.ema_production["food"] = 4.0
    assert emergence.relative_deficit(faction, "food") == pytest.approx(0.6)
    faction.ema_production["food"] = 25.0
    assert emergence.relative_deficit(faction, "food") == 0.0, "surplus => aucun déficit"


def test_a_building_with_no_deficit_resource_has_no_deficit(world):
    _cfg, _catalog, state, _rng = world
    assert emergence.relative_deficit(state.faction(0), None) == 0.0


def test_saturation_is_skipped_where_the_deficit_loop_suffices(world):
    """Pour les champs, `s(count)` est redondante : bâtir fait retomber l'envie."""
    _cfg, catalog, _state, _rng = world
    assert catalog.building("field").saturation_halflife is None
    assert emergence.saturation(10, None) == 1.0
    assert emergence.saturation(0, 2.0) == 1.0
    assert emergence.saturation(2, 2.0) == pytest.approx(0.5)


def test_the_field_desire_rises_with_the_food_deficit(world):
    """`criticality_k <= 1` doit faire réagir TÔT, avant la famine."""
    cfg, catalog, state, _rng = world
    faction = state.faction(0)
    spec = catalog.building("field")
    faction.ema_consumption["food"] = 20.0

    faction.ema_production["food"] = 20.0
    comfortable = emergence.building_desire(cfg, state, faction, spec)
    faction.ema_production["food"] = 10.0
    strained = emergence.building_desire(cfg, state, faction, spec)
    faction.ema_production["food"] = 1.0
    starving = emergence.building_desire(cfg, state, faction, spec)

    assert comfortable < strained < starving
    # Réaction précoce : à mi-déficit on a déjà franchi la moitié du chemin.
    assert strained > comfortable + (starving - comfortable) * 0.5


def test_relevant_traits_raise_the_building_desire(world):
    cfg, catalog, state, _rng = world
    faction = state.faction(0)
    spec = catalog.building("lab")
    neutral = emergence.building_desire(cfg, state, faction, spec)
    faction.traits["intelligence"] = 1.0
    assert emergence.building_desire(cfg, state, faction, spec) > neutral


def test_driver_value_rejects_an_unknown_key(world):
    _cfg, _catalog, state, _rng = world
    with pytest.raises(KeyError, match="driver d'envie inconnu"):
        emergence.driver_value(state, state.faction(0), "charisma")


def test_trade_relation_stays_symmetric(world):
    """Un embargo d'un côté doit couper les échanges des deux côtés (§6)."""
    _cfg, _catalog, state, _rng = world
    emergence.set_trade_relation(state, 0, 2, 0.7)
    assert state.faction(0).trade_relation[2] == 0.7
    assert state.faction(2).trade_relation[0] == 0.7
    emergence.set_trade_relation(state, 2, 0, -5.0)
    assert state.faction(0).trade_relation[2] == -1.0, "la relation doit être bornée"


def test_no_trade_below_the_threshold(world):
    cfg, _catalog, state, rng = world
    state.faction(0).food = 500.0
    emergence.set_trade_relation(state, 0, 1, float(cfg.trade.threshold) - 0.2)
    assert emergence.trade(cfg, state, rng) == []


# ── Appels et dérive (§4) ───────────────────────────────────────────────────

def test_a_call_saturates_as_the_trait_rises(world):
    """Le spam d'une seule action doit devenir inefficace — c'est ce qui force la
    diversification et empêche le runaway (§4)."""
    from controla.actions.apply import _call_trait

    cfg, _catalog, state, _rng = world
    faction = state.faction(0)
    steps = [_call_trait(cfg, faction, "warrior", 2.0) for _ in range(5)]
    assert steps == sorted(steps, reverse=True), "les Appels ne saturent pas"
    assert faction.traits["warrior"] <= float(cfg.traits.max)


def test_a_call_never_breaks_the_bounds(world):
    from controla.actions.apply import _call_trait

    cfg, _catalog, state, _rng = world
    faction = state.faction(0)
    for _ in range(200):
        _call_trait(cfg, faction, "warrior", 3.0)
    assert faction.traits["warrior"] <= float(cfg.traits.max)
    for _ in range(200):
        _call_trait(cfg, faction, "warrior", -3.0)
    assert faction.traits["warrior"] >= float(cfg.traits.min)


def test_an_unpressured_trait_drifts_back_towards_neutral_not_the_floor(world):
    """Correction de T1 : sans pression, un trait doit revenir à la BASELINE.

    Mapper l'absence de pression sur la borne basse ferait dériver tous les traits
    non sollicités vers -50 %, ce qui viderait le jeu de sa spécialisation.
    """
    cfg = base_config(traits={"drift_noise_sigma": 0.0})
    catalog = build_catalog(cfg)
    rng = KeyedRNG(3)
    state = create_state(cfg, catalog, rng)
    faction = state.faction(0)
    faction.traits["warrior"] = 0.8

    for turn in range(1, 25):
        state.turn = turn
        apply_drift(cfg, state, rng, {f.id: {} for f in state.factions})

    assert faction.traits["warrior"] == pytest.approx(0.0, abs=0.05), \
        "un trait non sollicité devrait revenir au neutre"
    assert faction.traits["warrior"] > float(cfg.traits.min) + 0.4


def test_repeated_wars_push_the_warrior_trait_up(world):
    """L'héritage émergent : le vécu façonne le peuple (§4)."""
    cfg = base_config(traits={"drift_noise_sigma": 0.0})
    catalog = build_catalog(cfg)
    rng = KeyedRNG(3)
    state = create_state(cfg, catalog, rng)
    faction = state.faction(0)

    for turn in range(1, 15):
        state.turn = turn
        apply_drift(cfg, state, rng, {0: {"fought": True}})
    assert faction.traits["warrior"] > 0.4


def test_drift_stays_within_bounds_under_maximum_pressure(world):
    cfg = base_config()
    catalog = build_catalog(cfg)
    rng = KeyedRNG(3)
    state = create_state(cfg, catalog, rng)
    for turn in range(1, 60):
        state.turn = turn
        apply_drift(cfg, state, rng, {0: {"fought": True, "built": True, "struck": True}})
    for value in state.faction(0).traits.values():
        assert float(cfg.traits.min) <= value <= float(cfg.traits.max)


# ── Économie ────────────────────────────────────────────────────────────────

def test_population_follows_a_logistic_curve(world):
    """Croissance rapide loin du plafond, stabilisation à K, déclin au-dessus (§8.4)."""
    cfg, _catalog, state, _rng = world
    faction = state.faction(0)
    capacity = economy.carrying_capacity(cfg, state, faction)

    faction.population = capacity * 0.1
    economy.grow_population(cfg, state)
    assert faction.population > capacity * 0.1

    faction.population = capacity * 2.0
    economy.grow_population(cfg, state)
    assert faction.population < capacity * 2.0, "au-dessus du plafond, la pop doit décliner"


def test_famine_kills_and_the_population_stays_non_negative(world):
    cfg, _catalog, state, _rng = world
    faction = state.faction(0)
    faction.hunger = 1.0
    for _ in range(200):
        economy.grow_population(cfg, state)
    assert faction.population >= 0.0


def test_consumption_is_reduced_by_the_resilient_trait(world):
    cfg, _catalog, state, _rng = world
    tough, frail = state.faction(0), state.faction(1)
    tough.traits["resilient"] = 1.0
    tough.food = frail.food = 1000.0
    tough.population = frail.population = 50.0

    report = economy.consume(cfg, state)
    assert report[tough.id]["need"] < report[frail.id]["need"]


def test_hunger_is_bounded_and_zero_when_fed(world):
    cfg, _catalog, state, _rng = world
    faction = state.faction(0)
    faction.food, faction.population = 10_000.0, 10.0
    economy.consume(cfg, state)
    assert faction.hunger == 0.0

    faction.food, faction.population = 0.0, 50.0
    economy.consume(cfg, state)
    assert faction.hunger == 1.0


def test_metal_appears_only_after_the_first_research_tier(world):
    """Spécialisation stricte : aucun terrain ne produit de métal (§2)."""
    cfg, catalog, state, rng = world
    faction = state.faction(0)
    assert faction.research_tier == 0
    economy.produce(cfg, catalog, state, rng)
    assert faction.metal == 0.0

    faction.research_tier = 1
    state.turn = 1
    economy.produce(cfg, catalog, state, rng)
    assert faction.metal > 0.0


def test_research_tiers_are_monotonic(world):
    """Un palier atteint n'est jamais reperdu — sinon le dieu ne peut plus planifier."""
    cfg, _catalog, state, _rng = world
    faction = state.faction(0)
    faction.research = float(cfg.research.tiers[2])
    economy.advance_research(cfg, state)
    assert faction.research_tier == 3

    faction.research = 0.0
    economy.advance_research(cfg, state)
    assert faction.research_tier == 3, "un palier a été reperdu"


def test_intervention_points_grow_logarithmically_and_are_capped(world):
    """Correction du plan : un scaling linéaire ferait du leader un dieu deux fois
    plus puissant, ce qui *crée* la stratégie dominante (§12)."""
    cfg, _catalog, state, _rng = world
    faction = state.faction(0)

    faction.population = 20.0
    economy.recharge_intervention_points(cfg, state)
    small = faction.intervention_points

    faction.population = 200.0
    economy.recharge_intervention_points(cfg, state)
    medium = faction.intervention_points

    faction.population = 10 ** 6
    economy.recharge_intervention_points(cfg, state)
    huge = faction.intervention_points

    assert small <= medium <= huge
    assert huge <= int(cfg.intervention.cap), "le plafond n'est pas respecté"
    assert (medium - small) < 10 * (huge - medium) + 5   # croissance très amortie


def test_blessing_raises_the_tile_yield(world):
    """Attribution : le dieu doit voir l'effet de son miracle."""
    from controla.state.effects import Modifier

    cfg, catalog, state, rng = world
    faction = state.faction(0)
    tile = state.tiles_of(faction.id)[0]
    stream = rng.stream(1, "test")

    plain = tile_total(cfg, catalog, state, faction, tile, stream)
    tile.modifiers["blessing:0"] = Modifier(
        float(cfg.miracles.blessing_bonus), until_turn=state.turn + 3, cause="test"
    )
    blessed = tile_total(cfg, catalog, state, faction, tile, stream)
    assert blessed > plain


def tile_total(cfg, catalog, state, faction, tile, stream) -> float:
    """Rendement total d'une case, bruit neutralisé par moyennage."""
    samples = [
        sum(economy.tile_yield(cfg, catalog, state, faction, tile, stream).values())
        for _ in range(60)
    ]
    return sum(samples) / len(samples)
