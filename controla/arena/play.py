"""Jouer une partie : la boucle AEC canonique.

C'est l'unique implémentation de « jouer une partie jusqu'au bout ». Le CLI, le
tournoi et les sweeps l'appellent tous — pas de seconde boucle qui divergerait.

L'arène assigne les policies aux sièges et rien de plus : l'environnement ignore
qui occupe un siège, la policy ignore qu'il y a un tournoi.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from controla.dynamics.scoring import final_report
from controla.env.controla_env import ControlaEnv, faction_id_of
from controla.policies.base import Policy


@dataclass
class MatchSpec:
    """Une partie : une seed, et une policy par siège."""

    seed: int
    seats: list[str]
    config_path: str | None = None
    journal_rng: bool = False
    metadata: dict = field(default_factory=dict)


def play_match(
    spec: MatchSpec,
    policies: list[Policy],
    config=None,
    max_steps: int = 200_000,
) -> dict:
    """Joue une partie et renvoie son résultat, prêt pour le stockage et le replay.

    `max_steps` est un garde-fou de terminaison : la partie s'arrête de toute façon
    au tour max, mais une policy pathologique pourrait multiplier les coups
    invalides — ce plafond garantit qu'aucun tournoi ne pend.
    """
    env = ControlaEnv(
        config=config, config_path=spec.config_path, journal_rng=spec.journal_rng
    )
    env.reset(seed=spec.seed)

    for agent, policy in zip(env.possible_agents, policies):
        policy.reset(faction_id_of(agent), spec.seed)

    by_agent = dict(zip(env.possible_agents, policies))
    started = time.perf_counter()
    steps = 0
    reports_seen = 0

    for agent in env.agent_iter(max_iter=max_steps):
        _observation, _reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            env.step(None)
            continue

        # L'arène complète `infos` avec ce dont une policy a besoin pour décoder
        # l'espace d'action. L'environnement, lui, n'en sait rien.
        info = {**info, "catalog": env.catalog, "index": env.index, "config": env.cfg}
        env.step(by_agent[agent].act(env.observe(agent), info))
        steps += 1

        while reports_seen < len(env.turn_reports):
            report = env.turn_reports[reports_seen]
            reports_seen += 1
            for policy in policies:
                policy.observe_turn(report.to_dict())

    result = {
        "seed": spec.seed,
        "seats": list(spec.seats),
        "turns_played": env.state_.turn,
        "steps": steps,
        "ranking": final_report(env.cfg, env.state_),
        "cumulative_rewards": {
            agent: round(value, 4)
            for agent, value in env._cumulative_rewards.items()
        },
        "invalid_actions": dict(env.invalid_actions),
        "history": [report.to_dict() for report in env.turn_reports],
        "wall_seconds": round(time.perf_counter() - started, 3),
        **spec.metadata,
    }

    # Les policies qui savent se rendre des comptes (LLM) versent leurs métriques :
    # taux d'action invalide, tours muets, latence. Sans elles, un modèle muet
    # passerait pour un modèle prudent.
    reports = {
        agent: policy.report()
        for agent, policy in by_agent.items()
        if hasattr(policy, "report")
    }
    if reports:
        result["policy_reports"] = reports
    if spec.journal_rng:
        result["rng_digest"] = env.rng.journal_digest()
    return result
