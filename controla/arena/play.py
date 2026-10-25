"""Jouer une partie : la boucle AEC canonique.

C'est l'unique implémentation de « jouer une partie jusqu'au bout ». Le CLI, le
tournoi et les sweeps l'appellent tous — pas de seconde boucle qui divergerait.

L'arène assigne les policies aux sièges et rien de plus : l'environnement ignore
qui occupe un siège, la policy ignore qu'il y a un tournoi.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
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
    # Joint à la partie de quoi la REJOUER visuellement : la carte et l'état du
    # plateau à chaque tour. Coupé par défaut — un sweep de 2 016 parties n'en
    # lit qu'un tableau de nombres, les lui écrire serait une centaine de
    # mégaoctets pour rien.
    snapshots: bool = False
    metadata: dict = field(default_factory=dict)


logger = logging.getLogger("controla.arena")


def parallel_rounds(cfg) -> bool:
    """La config demande-t-elle de planifier les rondes en parallèle ?"""
    try:
        return bool(cfg.llm.parallel_round)
    except AttributeError:
        return False


def plan_round(env, by_agent: dict, pool: ThreadPoolExecutor | None) -> dict | None:
    """Fait planifier la ronde à tous les sièges qui savent le faire d'avance.

    Seules les policies qui exposent `plan_ahead` (les LLM) sont concernées ;
    une policy scriptée décide en microsecondes et n'a rien à gagner. Les
    appels partent ensemble et l'on attend qu'ils soient TOUS rentrés : la ronde
    dure alors le plus lent des quatre, et non leur somme.

    L'ordre de jeu, lui, ne change pas — les coups restent soumis un siège
    après l'autre par la boucle AEC. Seule l'attente du modèle est mise en
    commun.

    Renvoie la durée de la ronde et celle de chaque siège — `None` s'il n'y
    avait rien à planifier (panel entièrement scripté). Le siège le plus lent
    est ce qui fixe la durée d'une ronde parallèle : c'est lui qu'il faut
    connaître pour dimensionner une campagne.
    """
    jobs = []
    for agent in env.agents:
        if env.terminations[agent] or env.truncations[agent]:
            continue
        # `get` : un siège sans policy (le joueur humain d'une session) n'a
        # rien à planifier.
        policy = by_agent.get(agent)
        if policy is None or not hasattr(policy, "plan_ahead"):
            continue
        info = {**env.infos[agent], "catalog": env.catalog, "index": env.index,
                "config": env.cfg}
        jobs.append((agent, policy, info))
    if not jobs:
        return None

    per_seat: dict[str, float] = {}

    def timed(agent, policy, info):
        started = time.perf_counter()
        try:
            policy.plan_ahead(info)
        finally:
            per_seat[agent] = round(time.perf_counter() - started, 3)

    started = time.perf_counter()
    if pool is None or len(jobs) < 2:
        for job in jobs:
            timed(*job)
    else:
        for future in [pool.submit(timed, *job) for job in jobs]:
            future.result()                           # propage une éventuelle erreur
    slowest = max(per_seat, key=per_seat.get)
    timing = {
        "turn": env.state_.turn,
        "seconds": round(time.perf_counter() - started, 3),
        "slowest": slowest,
        "per_seat": dict(sorted(per_seat.items())),
        "parallel": pool is not None and len(jobs) > 1,
    }
    logger.info(
        "ronde T%s planifiée en %.1f s — plus lent : %s (%.1f s)",
        timing["turn"], timing["seconds"], slowest, per_seat[slowest],
    )
    return timing


def _play_rounds(env, by_agent: dict, pool, max_steps: int,
                 timings: list[dict]) -> int:
    """La boucle AEC canonique. Renvoie le nombre de pas joués ; verse la durée
    de chaque ronde dans `timings`."""
    policies = list(by_agent.values())
    steps = 0
    reports_seen = 0

    def plan() -> None:
        timing = plan_round(env, by_agent, pool)
        if timing is not None:
            timings.append(timing)

    plan()                                            # première ronde

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

        new_round = reports_seen < len(env.turn_reports)
        while reports_seen < len(env.turn_reports):
            report = env.turn_reports[reports_seen]
            reports_seen += 1
            for policy in policies:
                policy.observe_turn(report.to_dict())
        # Une barrière vient de tomber : la ronde suivante commence. Les rapports
        # ont été versés AVANT, pour que chaque policy planifie en les ayant lus.
        if new_round and env.agents:
            plan()
    return steps


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
        config=config, config_path=spec.config_path, journal_rng=spec.journal_rng,
        snapshots=spec.snapshots,
    )
    env.reset(seed=spec.seed)

    world = None
    if spec.snapshots:
        from controla.replay.snapshot import board_snapshot, world_snapshot
        world = world_snapshot(env.cfg, env.catalog, env.state_)
        # Le tour 0 n'a pas de rapport de passe : sans cet instantané, la
        # restitution démarrerait après que le premier tour a déjà tout changé.
        world["initial_board"] = board_snapshot(env.cfg, env.catalog, env.state_)

    for agent, policy in zip(env.possible_agents, policies):
        policy.reset(faction_id_of(agent), spec.seed)

    by_agent = dict(zip(env.possible_agents, policies))
    started = time.perf_counter()
    pool = (
        ThreadPoolExecutor(max_workers=len(policies), thread_name_prefix="ronde")
        if parallel_rounds(env.cfg) else None
    )
    timings: list[dict] = []
    try:
        steps = _play_rounds(env, by_agent, pool, max_steps, timings)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

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
    if world is not None:
        result["world"] = world
    if timings:
        # Une entrée par ronde où un siège LLM a planifié : durée de la ronde,
        # siège le plus lent, durée de chaque siège. Écrit dans le JSONL, donc
        # disponible pour toute partie archivée.
        result["round_timings"] = timings

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
