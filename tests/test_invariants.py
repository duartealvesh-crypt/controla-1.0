"""Invariants structurels et terminaison, sur de nombreuses seeds.

Ces tests sont indépendants de l'équilibrage : ils doivent rester verts quelles que
soient les constantes. C'est ce qui permet de sweeper agressivement en tranche 5 —
un sweep qui les casse a cassé le moteur, pas l'équilibre.
"""

from __future__ import annotations

import pytest

from tests.helpers import (
    DEFAULT_SEATS,
    assert_invariants,
    base_config,
    run_scripted_match,
    run_with_checks,
)


@pytest.mark.parametrize("seed", range(12))
def test_invariants_hold_every_turn(seed):
    run_with_checks(seed, assert_invariants)


@pytest.mark.parametrize("seed", range(4))
def test_invariants_hold_under_aggressive_settings(seed):
    """Config poussée à la guerre : c'est là que les invariants cassent."""
    config = base_config(
        emergence={"war_threshold": 0.05, "war_call_boost": 2.0},
        combat={"attrition_loser": 0.9, "loot_fraction": 0.9},
    )
    run_with_checks(seed, assert_invariants, config=config)


@pytest.mark.parametrize("seed", range(3))
def test_invariants_hold_under_famine(seed):
    """Config affamante : population et Force doivent rester >= 0."""
    config = base_config(
        consumption={"food_per_capita": 4.0},
        production={"noise_sigma": 0.4},
    )
    run_with_checks(seed, assert_invariants, config=config)


@pytest.mark.parametrize("seed", range(6))
def test_game_always_terminates(seed):
    result = run_scripted_match(seed)
    assert result["turns_played"] <= 15
    assert result["steps"] < 100_000


def test_a_game_reaches_the_final_turn_or_a_sole_survivor():
    """Critère de sortie T1 : une partie va au bout."""
    result = run_scripted_match(3)
    survivors = [entry for entry in result["ranking"] if entry["alive"]]
    assert result["turns_played"] == 15 or len(survivors) <= 1


def test_ranking_is_complete_and_total():
    """OpenSkill a besoin d'un ordre d'arrivée complet : pas de match nul."""
    result = run_scripted_match(1)
    ranks = [entry["rank"] for entry in result["ranking"]]
    assert ranks == list(range(len(DEFAULT_SEATS)))
    assert len({entry["faction_id"] for entry in result["ranking"]}) == len(DEFAULT_SEATS)


def test_survivors_outrank_the_eliminated():
    for seed in range(10):
        result = run_scripted_match(seed)
        alive_flags = [entry["alive"] for entry in result["ranking"]]
        assert alive_flags == sorted(alive_flags, reverse=True), \
            f"seed {seed} : un éliminé devance un survivant"


# ── conservation des ressources ─────────────────────────────────────────────

def test_trade_conserves_resources():
    """Un échange déplace, il ne crée pas. Invariant explicite du plan."""
    from controla.dynamics.emergence import set_trade_relation, trade
    from controla.registry.catalog import build_catalog
    from controla.rng import KeyedRNG
    from controla.state.game_state import create_state
    from tests.helpers import total_resources

    cfg = base_config()
    rng = KeyedRNG(0)
    state = create_state(cfg, build_catalog(cfg), rng)
    state.faction(0).food = 100.0
    state.faction(1).wood = 80.0
    set_trade_relation(state, 0, 1, 1.0)

    before = total_resources(state)
    events = trade(cfg, state, rng)
    after = total_resources(state)

    assert events, "aucun échange déclenché au-dessus du seuil"
    for resource, amount in before.items():
        assert after[resource] == pytest.approx(amount, abs=1e-9), \
            f"le commerce a créé ou détruit du {resource}"


def test_loot_conserves_resources():
    """Le butin transfère, il ne crée pas."""
    from controla.dynamics.military import resolve_battle
    from controla.registry.catalog import build_catalog
    from controla.rng import KeyedRNG
    from controla.state.game_state import create_state
    from tests.helpers import total_resources

    cfg = base_config()
    rng = KeyedRNG(4)
    state = create_state(cfg, build_catalog(cfg), rng)
    attacker, defender = state.faction(0), state.faction(1)
    attacker.force, defender.force = 500.0, 5.0
    defender.food, defender.wood, defender.metal = 90.0, 60.0, 30.0
    target = next(tile.coord for tile in state.tiles_of(defender.id))

    before = total_resources(state)
    outcome = resolve_battle(
        cfg, state, attacker, defender, target,
        surprise=False, stream=rng.stream(1, "battle", 0),
    )
    after = total_resources(state)

    assert outcome.attacker_won, "le rapport de force rendait la victoire quasi certaine"
    for resource, amount in before.items():
        assert after[resource] == pytest.approx(amount, abs=1e-9), \
            f"le butin a créé ou détruit du {resource}"


def test_eliminated_faction_releases_its_territory():
    """Les cases d'un éliminé redeviennent neutres (§10)."""
    from controla.dynamics.development import _eliminate, TurnReport
    from controla.registry.catalog import build_catalog
    from controla.rng import KeyedRNG
    from controla.state.game_state import create_state

    cfg = base_config()
    state = create_state(cfg, build_catalog(cfg), KeyedRNG(0))
    state.turn = 5
    doomed = state.faction(2)
    doomed.population = 0.0
    next(iter(state.tiles_of(doomed.id))).building = "city"

    report = TurnReport(turn=5)
    _eliminate(cfg, state, report)

    assert not doomed.alive and doomed.eliminated_turn == 5
    assert state.tile_count(doomed.id) == 0
    assert report.eliminations == [doomed.id]
    assert all(
        tile.owner is not None or tile.building is None
        for tile in state.grid.all_tiles()
    ), "une case libérée a gardé son bâtiment"
