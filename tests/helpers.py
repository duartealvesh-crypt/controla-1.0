"""Outils de test partagés : runner de partie et invariants structurels.

Les invariants doivent tenir pour N'IMPORTE QUELLES constantes — c'est tout
l'intérêt du filet : ils restent verts pendant l'équilibrage (tranche 5), donc un
sweep qui les casse a cassé le moteur, pas l'équilibre.
"""

from __future__ import annotations

from controla.arena.play import MatchSpec, play_match
from controla.config.loader import load_config
from controla.env.controla_env import ControlaEnv, faction_id_of
from controla.policies.archetypes import build_policy

DEFAULT_SEATS = ("cooperator", "betrayer", "expansionist", "turtle")


def run_scripted_match(seed: int, seats=DEFAULT_SEATS, config=None, **spec_kwargs):
    """Joue une partie scriptée complète et renvoie son résultat."""
    return play_match(
        MatchSpec(seed=seed, seats=list(seats), **spec_kwargs),
        [build_policy(name) for name in seats],
        config=config,
    )


def run_with_checks(seed: int, per_turn, seats=DEFAULT_SEATS, config=None):
    """Joue une partie en appelant `per_turn(env)` après CHAQUE passe.

    On ne peut pas vérifier les invariants depuis le résultat final : un état
    intermédiaire invalide qui se résorbe passerait inaperçu.
    """
    env = ControlaEnv(config=config)
    env.reset(seed=seed)
    policies = [build_policy(name) for name in seats]
    for agent, policy in zip(env.possible_agents, policies):
        policy.reset(faction_id_of(agent), seed)
    by_agent = dict(zip(env.possible_agents, policies))

    per_turn(env)
    seen = 0
    for agent in env.agent_iter(max_iter=100_000):
        _obs, _reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            env.step(None)
            continue
        info = {**info, "catalog": env.catalog, "index": env.index, "config": env.cfg}
        env.step(by_agent[agent].act(env.observe(agent), info))
        if len(env.turn_reports) > seen:
            seen = len(env.turn_reports)
            per_turn(env)
    return env


def assert_invariants(env) -> None:
    """Invariants structurels, vrais à chaque tour et pour toute config."""
    cfg, state = env.cfg, env.state_
    turn = state.turn
    low, high = float(cfg.traits.min), float(cfg.traits.max)

    assert turn <= int(cfg.world.max_turns), f"la partie a dépassé le tour max (T{turn})"

    for faction in state.factions:
        tag = f"{faction.name} @T{turn}"

        # Stocks et stats jamais négatifs.
        for resource in ("food", "wood", "metal"):
            assert faction.stock(resource) >= 0, f"{tag} : {resource} < 0"
        assert faction.population >= 0, f"{tag} : population < 0"
        assert faction.research >= 0, f"{tag} : recherche < 0"
        assert faction.force >= 0, f"{tag} : Force < 0"
        assert faction.intervention_points >= 0, f"{tag} : points d'intervention < 0"
        assert 0.0 <= faction.hunger <= 1.0, f"{tag} : faim hors de [0, 1]"

        # Les traits restent bornés — sinon les Appels ou la dérive s'emballent.
        for key, value in faction.traits.items():
            assert low <= value <= high, f"{tag} : trait {key}={value} hors bornes"

        # La guerre est symétrique et jamais réflexive.
        assert faction.id not in faction.at_war_with, f"{tag} : en guerre contre soi"
        for enemy_id in faction.at_war_with:
            assert faction.id in state.faction(enemy_id).at_war_with, \
                f"{tag} : guerre non symétrique avec {enemy_id}"

        # La relation commerciale est symétrique (un embargo coupe des deux côtés).
        for other_id, value in faction.trade_relation.items():
            mirrored = state.faction(other_id).trade_relation.get(faction.id)
            assert mirrored == value, \
                f"{tag} : relation commerciale asymétrique avec {other_id}"

        # Une faction éliminée ne possède plus rien.
        if not faction.alive:
            assert state.tile_count(faction.id) == 0, \
                f"{tag} : éliminée mais possède encore du territoire"

    # Cohérence grille : un bâtiment implique un propriétaire.
    for tile in state.grid.all_tiles():
        if tile.building is not None:
            assert tile.owner is not None, \
                f"bâtiment sur une case sans propriétaire {tile.coord} @T{turn}"


def total_resources(state) -> dict[str, float]:
    """Somme des stocks sur toutes les factions — pour la conservation."""
    return {
        resource: sum(faction.stock(resource) for faction in state.factions)
        for resource in ("food", "wood", "metal")
    }


def base_config(**overrides):
    return load_config(overrides=overrides) if overrides else load_config()
