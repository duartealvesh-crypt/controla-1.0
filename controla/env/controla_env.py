"""`ControlaEnv` : l'arène, en PettingZoo AEC strict (spec §11).

Structure d'un tour
-------------------
Le « bundle » de §1 devient une **séquence de coups** : chaque dieu joue autant de
`step()` qu'il veut tant qu'il a des points, puis `end_turn` rend la main. Quand les
quatre ont rendu la main, la **barrière** tombe :

1. les actions bufferisées de la ronde sont appliquées, dans un ordre déterministe ;
2. la passe de développement s'exécute ;
3. rewards, terminaisons et nouvelles observations sont émis.

Rien n'est appliqué avant la barrière : un dieu qui joue en dernier ne voit pas les
coups des autres. C'est ce qui rend la résolution réellement simultanée.

Trois pièges AEC, traités explicitement
---------------------------------------
* **Agents morts** : PettingZoo exige `step(None)` sur un agent terminé, puis son
  retrait via `_was_dead_step`. C'est là que la plupart des implémentations AEC
  cassent, et c'est exactement le cas d'une faction éliminée.
* **`_cumulative_rewards`** : `rewards` porte le gain du dernier pas, le cumul est
  une structure distincte que `last()` renvoie.
* **Action illégale** : jamais de repli aléatoire (ça casserait le benchmark). Elle
  devient un no-op compté comme métrique de qualité d'instruction-following (§11.4).
"""

from __future__ import annotations

import functools

from pettingzoo.utils.env import AECEnv

from controla.actions.apply import apply_buffered, submit
from controla.actions.legality import ActionIndex, is_legal
from controla.config.loader import load_config
from controla.dynamics.development import run_development_pass
from controla.dynamics.scoring import is_over
from controla.env import observation as obs_module
from controla.env.reward import score_snapshot, shaping_rewards, terminal_rewards
from controla.registry.catalog import build_catalog
from controla.registry.descriptors import ActionKind
from controla.rng import KeyedRNG
from controla.state.game_state import create_state

AGENT_PREFIX = "god_"


def agent_name(faction_id: int) -> str:
    return f"{AGENT_PREFIX}{faction_id}"


def faction_id_of(agent: str) -> int:
    return int(agent[len(AGENT_PREFIX):])


class ControlaEnv(AECEnv):
    metadata = {"render_modes": ["human", "ansi"], "name": "controla/GodGame-v0",
                "is_parallelizable": False}

    def __init__(self, config=None, config_path=None, render_mode: str | None = None,
                 journal_rng: bool = False, snapshots: bool = False):
        super().__init__()
        self.cfg = config if config is not None else load_config(config_path)
        self.catalog = build_catalog(self.cfg)
        self.render_mode = render_mode
        self._journal_rng = journal_rng
        # Instantanés de plateau pour la restitution (§14.6). Optionnels, et
        # coupés par défaut : un sweep joue des milliers de parties dont il ne
        # lit qu'un tableau de nombres, et n'a que faire de leurs cartes.
        self._snapshots = snapshots

        count = int(self.cfg.world.num_factions)
        self.possible_agents = [agent_name(index) for index in range(count)]
        self.index = ActionIndex(
            self.catalog, int(self.cfg.world.grid_size), count
        )
        self.state_ = None
        self.rng = None
        self._buffer = []
        self._submissions = {}
        self._yielded = set()
        self._seq = 0
        self._score_before = {}
        self.turn_reports = []
        self.invalid_actions = {}

    # ── espaces ─────────────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return obs_module.observation_space(self.cfg, self.catalog, self.index)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return obs_module.action_space(self.cfg, self.index)

    # ── reset ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self.seed_value = 0 if seed is None else int(seed)
        self.rng = KeyedRNG(self.seed_value, journal_enabled=self._journal_rng)
        self.state_ = create_state(self.cfg, self.catalog, self.rng)

        self.agents = list(self.possible_agents)
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._buffer = []
        self._submissions = {agent: 0 for agent in self.agents}
        self._yielded = set()
        self._seq = 0
        self.turn_reports = []
        self.invalid_actions = {agent: 0 for agent in self.agents}
        # Utilisé par `_was_dead_step` pour reprendre la main après avoir purgé la
        # chaîne des agents morts. PettingZoo le lit en `getattr` ; on l'initialise
        # explicitement pour que l'état de l'env soit complet dès le reset.
        self._skip_agent_selection = None

        # Le tour 0 charge les points d'intervention : sans cela, personne ne peut
        # rien jouer à la première ronde.
        from controla.dynamics import economy, military
        economy.recharge_intervention_points(self.cfg, self.state_)
        military.refresh_force(self.cfg, self.state_)

        self._score_before = score_snapshot(self.cfg, self.state_)
        self.agent_selection = self.agents[0]
        self._refresh_infos()
        return None

    # ── observation ─────────────────────────────────────────────────────────

    def observe(self, agent: str) -> dict:
        return obs_module.observe(
            self.cfg, self.catalog, self.state_, self.index, faction_id_of(agent)
        )

    def _refresh_info(self, agent: str) -> None:
        """Recalcule la vue d'UN agent.

        À faire à chaque changement de sélection, et pas seulement à la barrière :
        dépenser un point change le masque, et servir une vue périmée ferait
        proposer à la policy des coups qu'elle ne peut plus payer — ils seraient
        comptés invalides alors que c'est l'environnement qui a menti.
        """
        self.infos[agent] = {
            "view": obs_module.build_view(
                self.cfg, self.catalog, self.state_, self.index, faction_id_of(agent)
            ),
            "invalid_actions": self.invalid_actions[agent],
        }

    def _refresh_infos(self) -> None:
        for agent in self.agents:
            self._refresh_info(agent)

    # ── step ────────────────────────────────────────────────────────────────

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            # Protocole AEC : un agent terminé reçoit `None` et sort de la liste.
            self._was_dead_step(action)
            # `_was_dead_step` mémorise dans `_skip_agent_selection` l'agent qui
            # jouait au moment de la purge, pour y revenir ensuite. Quand c'est
            # NOUS qui avons pointé la sélection sur un mort (à la barrière), il
            # y restaure donc un agent déjà retiré. PettingZoo attend que l'env
            # reprenne la main derrière — sinon `last()` lève un KeyError.
            if self.agents and self.agent_selection not in self.agents:
                self._skip_agent_selection = None
                self.agent_selection = self._next_selection()
            return

        self._cumulative_rewards[agent] = 0.0
        yielded = self._apply_submission(agent, action)

        if yielded:
            self._yielded.add(agent)

        if self._yielded >= set(self.agents):
            self._barrier()
        else:
            self._advance_selection()
            # La vue du prochain à jouer doit refléter les points déjà dépensés.
            self._refresh_info(self.agent_selection)

        self._accumulate_rewards()

    def _apply_submission(self, agent: str, action) -> bool:
        """Enregistre un coup. Renvoie True si le dieu a rendu la main.

        Une action illégale est un **no-op compté** — jamais un repli aléatoire, qui
        fausserait le benchmark en créditant au modèle un coup qu'il n'a pas choisi.
        """
        faction_id = faction_id_of(agent)
        self._submissions[agent] += 1

        # `int` = le contrat RL déclaré ; `dict` = le sur-ensemble textuel.
        if isinstance(action, dict):
            raw_index = int(action.get("action", self.index.end_turn_index))
            argument = str(action.get("argument", "") or "")
        else:
            raw_index, argument = int(action), ""

        if self._submissions[agent] >= int(self.cfg.intervention.max_submissions_per_turn):
            return True     # plafond atteint : le tour se termine d'office

        if not 0 <= raw_index < len(self.index):
            self.invalid_actions[agent] += 1
            return False

        entry = self.index[raw_index]
        if entry.spec.kind is ActionKind.CONTROL:
            return True

        if not is_legal(self.cfg, self.state_, faction_id, entry):
            self.invalid_actions[agent] += 1
            return False

        self._buffer.append(submit(
            self.cfg, self.state_, faction_id, entry, self._seq, argument=argument,
        ))
        self._seq += 1
        return False

    def _advance_selection(self) -> None:
        """Passe au prochain dieu qui n'a pas encore rendu la main."""
        order = self.agents
        start = order.index(self.agent_selection)
        for offset in range(1, len(order) + 1):
            candidate = order[(start + offset) % len(order)]
            if candidate not in self._yielded:
                self.agent_selection = candidate
                return
        self.agent_selection = order[0]

    # ── barrière de ronde ───────────────────────────────────────────────────

    def _barrier(self) -> None:
        """Applique la ronde, exécute la passe, émet rewards et terminaisons."""
        events = apply_buffered(self.cfg, self.catalog, self.state_, self._buffer)
        self._buffer.clear()
        self._seq = 0

        report = run_development_pass(self.cfg, self.catalog, self.state_, self.rng)
        report.events = events + report.events
        if self._snapshots:
            # Posé ICI et pas ailleurs : c'est le seul instant où l'état est
            # exactement « fin du tour N », après la passe et avant que la ronde
            # suivante ne bufferise quoi que ce soit.
            from controla.replay.snapshot import board_snapshot
            report.board = board_snapshot(self.cfg, self.catalog, self.state_)
        self.turn_reports.append(report)

        after = score_snapshot(self.cfg, self.state_)
        shaped = shaping_rewards(self.cfg, self._score_before, after)
        self._score_before = after
        for agent in self.agents:
            self.rewards[agent] = shaped.get(faction_id_of(agent), 0.0)

        terminated, truncated = is_over(self.cfg, self.state_)
        if terminated or truncated:
            final = terminal_rewards(self.cfg, self.state_)
            for agent in self.agents:
                self.rewards[agent] += final.get(faction_id_of(agent), 0.0)
            self.terminations = {agent: terminated for agent in self.agents}
            self.truncations = {agent: truncated for agent in self.agents}
        else:
            # Une faction éliminée se termine seule ; la partie continue.
            for agent in self.agents:
                self.terminations[agent] = not self.state_.faction(
                    faction_id_of(agent)
                ).alive

        self._yielded.clear()
        self._submissions = {agent: 0 for agent in self.agents}
        self._skip_agent_selection = None
        self._refresh_infos()
        self.agent_selection = self._next_selection()

    def _next_selection(self) -> str:
        """Les agents morts passent en premier.

        PettingZoo ne retire un agent terminé que lorsqu'il est SÉLECTIONNÉ et
        reçoit `step(None)`. Si on ne les sélectionnait jamais, ils resteraient
        indéfiniment dans `self.agents` et `agent_iter` ne se terminerait pas —
        c'est le piège classique des implémentations AEC.
        """
        if not self.agents:
            return self.possible_agents[0]
        for agent in self.agents:
            if self.terminations[agent] or self.truncations[agent]:
                return agent
        return self.agents[0]

    # ── rendu ───────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode is None:
            return None
        from controla.replay.render import render_state
        text = render_state(self.cfg, self.catalog, self.state_)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self):
        return None
