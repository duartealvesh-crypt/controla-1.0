"""Partie jouable : un `ControlaEnv` piloté requête par requête.

Pourquoi ne pas réutiliser `play_match`
---------------------------------------
`play_match` est une boucle qui va au bout : elle appelle chaque policy, encaisse
son coup, recommence. C'est le bon modèle pour un banc — et le mauvais pour un
humain, dont la « policy » est un aller-retour HTTP de durée indéterminée. Le
faire entrer dans `play_match` demanderait une policy qui bloque sur une file et
un thread pour la nourrir : deux mécanismes, un interblocage possible, et une
seconde façon de jouer une partie qui finirait par diverger de la première.

On inverse donc le contrôle. La session avance **jusqu'à ce que ce soit au tour
de l'humain**, puis rend la main. Le navigateur envoie un coup, la session
avance de nouveau. Aucun thread, aucune file, et la boucle reste la boucle AEC
canonique — les mêmes appels dans le même ordre que `play_match`.

Ce que ça ne fait pas
---------------------
Cette session ne sert PAS à mesurer. Une partie jouée par un humain n'entre dans
aucun classement : il n'a ni les mêmes contraintes de temps, ni le même
brouillard sur ce qu'il a lu à l'écran. Elle sert à sentir l'équilibrage — voir
de l'intérieur pourquoi une voie paie ou non.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from concurrent.futures import ThreadPoolExecutor

from controla.arena.play import parallel_rounds, plan_round
from controla.dynamics.scoring import final_report
from controla.env.controla_env import ControlaEnv, agent_name, faction_id_of
from controla.policies.factory import build_policies
from controla.replay.snapshot import board_snapshot, world_snapshot

# Nom réservé qui, dans une liste de sièges, désigne le joueur humain.
HUMAN_SEAT = "human"

# Garde-fou de terminaison, comme `play_match.max_steps` : une policy
# pathologique ne doit jamais faire pendre une requête HTTP.
MAX_STEPS = 100_000


@dataclass
class SessionSpec:
    seed: int
    seats: list[str]
    human_seat: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.human_seat < len(self.seats):
            raise ValueError(
                f"siège humain {self.human_seat} hors des {len(self.seats)} sièges"
            )


@dataclass
class Session:
    """Une partie en cours, pilotée depuis le navigateur."""

    spec: SessionSpec
    env: ControlaEnv
    policies: dict[str, object]
    world: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    history: list[dict] = field(default_factory=list)
    _reports_seen: int = 0
    _planned_round: int = -1

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def create(cls, spec: SessionSpec, cfg, catalog, backends: dict | None = None) -> "Session":
        env = ControlaEnv(config=cfg, snapshots=True)
        env.reset(seed=spec.seed)

        # Les sièges IA seulement : le siège humain n'a pas de policy, c'est tout
        # l'intérêt. On construit la liste dans l'ordre pour que `build_policies`
        # voie exactement les noms qu'il connaît.
        ai_seats = [
            name for index, name in enumerate(spec.seats) if index != spec.human_seat
        ]
        built = build_policies(ai_seats, cfg, catalog, backends if backends is not None else {})

        policies: dict[str, object] = {}
        cursor = 0
        for index in range(len(spec.seats)):
            if index == spec.human_seat:
                continue
            policy = built[cursor]
            cursor += 1
            policy.reset(index, spec.seed)
            policies[agent_name(index)] = policy

        world = world_snapshot(env.cfg, env.catalog, env.state_)
        # Comme `play_match` : sans l'état initial, le replay d'une partie jouée
        # à la main démarrerait au tour 1, une image de moins que les autres.
        world["initial_board"] = board_snapshot(env.cfg, env.catalog, env.state_)
        return cls(spec=spec, env=env, policies=policies, world=world)

    # ── état ─────────────────────────────────────────────────────────────────

    @property
    def human_agent(self) -> str:
        return agent_name(self.spec.human_seat)

    @property
    def finished(self) -> bool:
        return not self.env.agents

    @property
    def waiting_for_human(self) -> bool:
        return (
            not self.finished
            and self.env.agent_selection == self.human_agent
            and self.human_agent in self.env.agents
        )

    # ── avancement ───────────────────────────────────────────────────────────

    def advance(self) -> None:
        """Fait jouer les IA jusqu'au tour de l'humain, ou jusqu'à la fin."""
        env = self.env
        steps = 0
        while env.agents and steps < MAX_STEPS:
            agent = env.agent_selection
            _observation, _reward, terminated, truncated, info = env.last()

            if terminated or truncated:
                env.step(None)
                self._drain_reports()
                continue

            if agent == self.human_agent:
                return                                 # à lui de jouer

            self._plan_round()
            info = {**info, "catalog": env.catalog, "index": env.index, "config": env.cfg}
            env.step(self.policies[agent].act(env.observe(agent), info))
            steps += 1
            self._drain_reports()
        self._drain_reports()

    def _plan_round(self) -> None:
        """Fait planifier ensemble les adversaires LLM, une fois par ronde.

        Même mécanisme que `play_match` : le joueur humain n'attend plus trois
        appels l'un après l'autre avant de récupérer la main.
        """
        turn = self.env.state_.turn
        if turn == self._planned_round:
            return
        self._planned_round = turn
        if not parallel_rounds(self.env.cfg):
            return                                     # chaque IA planifiera à son tour
        with ThreadPoolExecutor(max_workers=max(1, len(self.policies))) as pool:
            plan_round(self.env, self.policies, pool)

    def submit(self, index: int, argument: str = "") -> None:
        """Enregistre un coup humain, puis laisse les IA reprendre la main."""
        if not self.waiting_for_human:
            raise RuntimeError("ce n'est pas au joueur humain de jouer")
        self.env.step({"action": int(index), "argument": str(argument or "")})
        self._drain_reports()
        self.advance()

    def _drain_reports(self) -> None:
        """Verse les rapports de tour nouveaux aux policies et à l'historique.

        Les policies scriptées lisent ce flux pour réagir (répondre à une main
        tendue, rendre un coup). Le sauter les rendrait sourdes, et le panel ne
        se comporterait pas comme dans une campagne.
        """
        while self._reports_seen < len(self.env.turn_reports):
            report = self.env.turn_reports[self._reports_seen].to_dict()
            self._reports_seen += 1
            self.history.append(report)
            for policy in self.policies.values():
                policy.observe_turn(report)

    # ── restitution ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Tout ce dont le navigateur a besoin pour peindre l'écran."""
        env = self.env
        payload: dict = {
            "id": self.id,
            "seed": self.spec.seed,
            "seats": list(self.spec.seats),
            "human_seat": self.spec.human_seat,
            # Le monde immuable voyage à chaque rafraîchissement. Il pèse ~1 ko
            # sur une grille de 11 ; le mettre en cache côté navigateur ferait
            # gagner cela et coûterait un chemin de code où l'écran peut se
            # désynchroniser du serveur.
            "world": self.world,
            "turn": env.state_.turn,
            "max_turns": int(env.cfg.world.max_turns),
            "finished": self.finished,
            "waiting_for_human": self.waiting_for_human,
            "board": board_snapshot(env.cfg, env.catalog, env.state_),
            "history": self.history,
        }
        if self.waiting_for_human:
            # `infos` est rafraîchi à chaque changement de sélection : la vue
            # servie est donc à jour des points déjà dépensés dans le tour.
            payload["view"] = env.infos[self.human_agent]["view"]
        if self.finished:
            payload["ranking"] = final_report(env.cfg, env.state_)
        return payload

    def result(self) -> dict:
        """La partie sous la forme que produit `play_match`, donc rejouable.

        Même dictionnaire, même clés : la visionneuse ne fait aucune différence
        entre une partie de banc et une partie jouée à la main.
        """
        env = self.env
        return {
            "seed": self.spec.seed,
            "seats": list(self.spec.seats),
            "turns_played": env.state_.turn,
            "steps": 0,
            "ranking": final_report(env.cfg, env.state_),
            "cumulative_rewards": {
                agent: round(value, 4)
                for agent, value in env._cumulative_rewards.items()
            },
            "invalid_actions": dict(env.invalid_actions),
            "history": self.history,
            "wall_seconds": round(time.time() - self.started_at, 3),
            "world": {**self.world, "human_seat": self.spec.human_seat},
            "policy_reports": {
                agent: policy.report()
                for agent, policy in self.policies.items()
                if hasattr(policy, "report")
            },
        }


class SessionStore:
    """Les sessions vivantes, en mémoire.

    En mémoire et pas sur disque : une partie jouable n'est pas une mesure, elle
    n'a donc pas à survivre au serveur. Ce qui mérite d'être gardé est sauvé
    explicitement, par `result()`, dans le même format que le reste.
    """

    def __init__(self, limit: int = 8) -> None:
        self._sessions: dict[str, Session] = {}
        self._limit = limit

    def add(self, session: Session) -> Session:
        # Plafond dur : un onglet oublié ne doit pas faire croître la mémoire
        # d'un serveur qui tourne sur le poste de quelqu'un.
        while len(self._sessions) >= self._limit:
            oldest = min(self._sessions.values(), key=lambda item: item.started_at)
            del self._sessions[oldest.id]
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
