"""Conformité PettingZoo AEC.

`api_test` et `seed_test` sont fournis par PettingZoo : ils valident gratuitement
tout le protocole (espaces, `last()`, cumul des rewards, protocole des agents morts,
reproductibilité). C'est le critère de sortie principal de la tranche 1 — si l'arène
n'est pas conforme, aucune bibliothèque RL ne pourra s'y brancher.
"""

from __future__ import annotations

import pytest
from pettingzoo.test import api_test, seed_test

from controla.config.loader import load_config
from controla.env.controla_env import ControlaEnv, agent_name, faction_id_of


def make_env(**overrides):
    cfg = load_config(overrides=overrides) if overrides else None
    return ControlaEnv(config=cfg)


def test_pettingzoo_api_conformance():
    """Le test de conformité officiel, sur une partie complète."""
    api_test(make_env(), num_cycles=60, verbose_progress=False)


def test_pettingzoo_seed_conformance():
    """Deux env de même seed produisent des trajectoires identiques."""
    seed_test(make_env, num_cycles=40)


def test_agent_naming_roundtrips():
    for faction_id in range(4):
        assert faction_id_of(agent_name(faction_id)) == faction_id


def test_reset_returns_a_symmetric_start():
    env = make_env()
    env.reset(seed=3)
    state = env.state_
    assert state.turn == 0
    assert len(env.agents) == int(env.cfg.world.num_factions)
    stocks = {
        (f.food, f.wood, f.metal, f.population, f.research) for f in state.factions
    }
    assert len(stocks) == 1, "les départs ne sont pas identiques"
    assert all(
        all(value == 0.0 for value in f.traits.values()) for f in state.factions
    ), "les traits de départ ne sont pas neutres"


def test_intervention_points_are_loaded_at_reset():
    """Sans cela, personne ne peut jouer à la première ronde."""
    env = make_env()
    env.reset(seed=1)
    assert all(f.intervention_points > 0 for f in env.state_.factions)


def test_observation_matches_its_space():
    env = make_env()
    env.reset(seed=5)
    for agent in env.agents:
        space = env.observation_space(agent)
        assert space.contains(env.observe(agent))


def test_action_mask_agrees_with_the_legality_source():
    """Le masque et l'énumération du prompt viennent de la même fonction — sinon un
    LLM se verrait proposer un coup que l'env refuserait."""
    env = make_env()
    env.reset(seed=5)
    for agent in env.agents:
        mask = env.observe(agent)["action_mask"]
        listed = {item["index"] for item in env.infos[agent]["view"]["legal_actions"]}
        assert {i for i, bit in enumerate(mask) if bit} == listed


def test_end_turn_is_always_legal():
    """Sans elle, un dieu sans point bloquerait la partie."""
    env = make_env()
    env.reset(seed=5)
    for agent in env.agents:
        assert env.observe(agent)["action_mask"][env.index.end_turn_index] == 1


def test_illegal_action_is_a_counted_noop_never_a_random_fallback():
    """§11.4 : jamais d'action aléatoire de repli — ça casserait le benchmark.

    Un coup illégal ne doit rien appliquer et être compté comme métrique de qualité
    d'instruction-following.
    """
    env = make_env()
    env.reset(seed=5)
    agent = env.agent_selection
    faction = env.state_.faction(faction_id_of(agent))
    before_points = faction.intervention_points
    before_traits = dict(faction.traits)

    # Une action ciblant une case ennemie alors qu'elle est neutre : hors masque.
    illegal = next(
        entry.index for entry in env.index.entries
        if env.observe(agent)["action_mask"][entry.index] == 0
    )
    env.step(illegal)

    assert env.invalid_actions[agent] == 1
    assert faction.intervention_points == before_points, "un coup illégal a coûté des points"
    assert faction.traits == before_traits, "un coup illégal a eu un effet"


def test_out_of_range_index_is_rejected_without_crashing():
    env = make_env()
    env.reset(seed=5)
    agent = env.agent_selection
    env.step(10 ** 9)
    assert env.invalid_actions[agent] == 1


def test_a_turn_ends_even_under_a_flood_of_invalid_actions():
    """Le plafond de soumissions garantit la terminaison face à une policy pathologique."""
    env = make_env()
    env.reset(seed=5)
    agent = env.agent_selection
    illegal = next(
        entry.index for entry in env.index.entries
        if env.observe(agent)["action_mask"][entry.index] == 0
    )
    for _ in range(int(env.cfg.intervention.max_submissions_per_turn) + 2):
        if env.agent_selection != agent:
            break
        env.step(illegal)
    assert env.agent_selection != agent, "le dieu n'a jamais rendu la main"


def test_actions_are_buffered_until_the_barrier():
    """Résolution simultanée (§1) : un dieu qui joue tôt ne doit rien révéler.

    On vérifie que l'effet n'est PAS appliqué à la soumission — sinon un dieu jouant
    en dernier dans la ronde verrait les coups des autres.
    """
    env = make_env()
    env.reset(seed=5)
    agent = env.agent_selection
    faction = env.state_.faction(faction_id_of(agent))

    influence = next(
        entry for entry in env.index.entries
        if entry.spec.effects
        and entry.spec.effects[0].kind == "trait_call"
        and env.observe(agent)["action_mask"][entry.index] == 1
    )
    trait = influence.spec.effects[0].trait
    before = faction.traits[trait]

    env.step(influence.index)
    assert faction.traits[trait] == before, "l'effet a fui avant la barrière"
    assert faction.intervention_points < int(env.cfg.intervention.cap) + 1


@pytest.mark.parametrize("seed", [0, 7])
def test_game_reaches_the_barrier_and_advances_a_turn(seed):
    env = make_env()
    env.reset(seed=seed)
    for _ in range(len(env.agents)):
        env.step(env.index.end_turn_index)
    assert env.state_.turn == 1
    assert len(env.turn_reports) == 1
