"""Déterminisme et équité de l'aléa.

Deux exigences distinctes :

* **Reproductibilité** — même seed ⇒ même partie, jusqu'à la trajectoire d'aléa.
* **Équité** — le nombre d'actions d'un dieu ne doit PAS décaler l'aléa subi par
  les autres. Sans cela, un modèle bavard changerait les récoltes de son voisin et
  le classement mesurerait la chance (§11).
"""

from __future__ import annotations

import pytest

from controla.env.controla_env import ControlaEnv, faction_id_of
from controla.policies.archetypes import build_policy
from tests.helpers import DEFAULT_SEATS, base_config, run_scripted_match


def _digest(result: dict) -> tuple:
    """Empreinte serrée d'une partie : classement, scores, et trajectoire d'aléa."""
    return (
        result["turns_played"],
        tuple(
            (e["faction_id"], e["rank"], e["score"], tuple(sorted(e["components"].items())))
            for e in result["ranking"]
        ),
        result.get("rng_digest"),
    )


@pytest.mark.parametrize("seed", [0, 7, 20260805])
def test_same_seed_replays_identically(seed):
    first = run_scripted_match(seed, journal_rng=True)
    second = run_scripted_match(seed, journal_rng=True)
    assert _digest(first) == _digest(second)


def test_the_rng_journal_itself_is_identical():
    """Comparer la TRAJECTOIRE d'aléa, pas seulement l'état final : deux parties
    peuvent finir pareil en ayant divergé en chemin."""
    first = run_scripted_match(11, journal_rng=True)
    second = run_scripted_match(11, journal_rng=True)
    assert first["rng_digest"] == second["rng_digest"]
    assert first["rng_digest"] is not None


def test_different_seeds_diverge():
    assert _digest(run_scripted_match(1)) != _digest(run_scripted_match(2))


def test_history_is_identical_across_replays():
    first = run_scripted_match(5)
    second = run_scripted_match(5)
    assert first["history"] == second["history"]


def test_journalling_does_not_change_the_game():
    """Activer l'audit ne doit rien changer au déroulement."""
    quiet = run_scripted_match(9, journal_rng=False)
    loud = run_scripted_match(9, journal_rng=True)
    assert quiet["ranking"] == loud["ranking"]
    assert quiet["history"] == loud["history"]


# ── équité de l'aléa ────────────────────────────────────────────────────────

def _run_with_extra_noops(seed: int, noisy_agent: str, extra: int) -> dict:
    """Joue une partie où un dieu soumet `extra` coups invalides supplémentaires.

    Les coups invalides ne coûtent rien et ne changent donc PAS l'état du jeu —
    mais s'ils consommaient de l'aléa partagé, ils décaleraient tout le reste.
    """
    env = ControlaEnv()
    env.reset(seed=seed)
    policies = [build_policy(name) for name in DEFAULT_SEATS]
    for agent, policy in zip(env.possible_agents, policies):
        policy.reset(faction_id_of(agent), seed)
    by_agent = dict(zip(env.possible_agents, policies))

    illegal = next(
        entry.index for entry in env.index.entries
        if env.observe(env.agent_selection)["action_mask"][entry.index] == 0
    )
    budget = {agent: (extra if agent == noisy_agent else 0) for agent in env.agents}

    for agent in env.agent_iter(max_iter=200_000):
        _obs, _reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            env.step(None)
            continue
        if budget.get(agent, 0) > 0:
            budget[agent] -= 1
            env.step(illegal)
            continue
        info = {**info, "catalog": env.catalog, "index": env.index, "config": env.cfg}
        env.step(by_agent[agent].act(env.observe(agent), info))

    return {
        "population": {f.id: round(f.population, 6) for f in env.state_.factions},
        "tiles": {f.id: env.state_.tile_count(f.id) for f in env.state_.factions},
    }


def test_extra_submissions_by_one_god_do_not_shift_the_others():
    """LE test qui prouve `KeyedRNG`.

    Un dieu qui soumet trois coups invalides de plus ne doit modifier ni sa propre
    économie ni celle des autres : l'aléa est keyé par `(seed, tour, purpose,
    faction)`, jamais par l'ordre d'appel.
    """
    quiet = _run_with_extra_noops(3, noisy_agent="god_0", extra=0)
    noisy = _run_with_extra_noops(3, noisy_agent="god_0", extra=3)
    assert noisy == quiet


def test_rng_streams_are_independent_across_factions():
    """Consommer beaucoup d'aléa pour une faction ne change rien pour les autres."""
    from controla.rng import KeyedRNG

    reference = KeyedRNG(77).stream(4, "production", 2).random()
    busy = KeyedRNG(77)
    for _ in range(200):
        busy.stream(4, "production", 0).random()
        busy.stream(4, "build", 1).random()
    assert busy.stream(4, "production", 2).random() == reference


def test_worldgen_depends_only_on_the_seed():
    """La carte ne doit pas dépendre de ce qui se passe ensuite."""
    from controla.registry.catalog import build_catalog
    from controla.rng import KeyedRNG
    from controla.state.game_state import create_state

    cfg = base_config()
    catalog = build_catalog(cfg)

    plain = create_state(cfg, catalog, KeyedRNG(31))
    rng = KeyedRNG(31)
    for _ in range(50):                       # bruit consommé AVANT la génération
        rng.stream(9, "unrelated", 3).random()
    disturbed = create_state(cfg, catalog, rng)

    assert [t.terrain for t in plain.grid.all_tiles()] == \
           [t.terrain for t in disturbed.grid.all_tiles()]
