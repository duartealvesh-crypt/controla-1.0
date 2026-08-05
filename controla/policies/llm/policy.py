"""`LLMPolicy` : un dieu piloté par un modèle de langage.

Un appel par tour, pas par coup. Le modèle planifie tout son tour d'un coup — c'est
le « bundle » de §1 — puis la policy vide ce plan coup par coup dans l'arène. Le
budget tient ainsi à ~60 appels par partie plutôt qu'à plusieurs centaines.

Discipline face aux sorties invalides (§11.4), dans cet ordre :

1. Le prompt **énumère** les coups légaux, avec leurs index.
2. La sortie est **structurée** (tool calling, ou schéma JSON annexé).
3. Une sortie invalide déclenche une relance **avec l'erreur précise**, N fois.
4. Sinon : **no-op**. Jamais d'action aléatoire de repli — cela créditerait au
   modèle un coup qu'il n'a pas choisi, et le classement deviendrait faux.
5. Le taux d'action invalide est **loggé comme métrique** de qualité
   d'instruction-following, pas balayé sous le tapis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from controla.policies.base import Policy, action
from controla.policies.llm.backend import LLMBackend
from controla.policies.llm.prompt import PLAN_TOOL, build_system, build_user

logger = logging.getLogger("controla.llm.policy")


@dataclass
class PolicyStats:
    """Ce qu'il faut savoir pour ne pas confondre un modèle prudent et un modèle muet."""

    turns: int = 0
    planned_moves: int = 0
    rejected_moves: int = 0        # index hors de la liste légale au moment de jouer
    empty_plans: int = 0           # le modèle a répondu, mais n'a rien joué
    mute_turns: int = 0            # aucune réponse exploitable après relances
    retries: int = 0
    reasonings: list[dict] = field(default_factory=list)

    @property
    def move_accuracy(self) -> float | None:
        """Part des coups proposés qui étaient réellement jouables."""
        if not self.planned_moves:
            return None
        return 1.0 - self.rejected_moves / self.planned_moves

    def to_dict(self) -> dict:
        return {
            "turns": self.turns,
            "planned_moves": self.planned_moves,
            "rejected_moves": self.rejected_moves,
            "move_accuracy": (
                None if self.move_accuracy is None else round(self.move_accuracy, 4)
            ),
            "empty_plans": self.empty_plans,
            "mute_turns": self.mute_turns,
            "retries": self.retries,
        }


class LLMPolicy(Policy):
    """Policy adossée à un `LLMBackend`."""

    def __init__(self, backend: LLMBackend, cfg, catalog, keep_reasoning: bool = True):
        self.backend = backend
        self.name = backend.model
        self.cfg = cfg
        self.catalog = catalog
        self.keep_reasoning = keep_reasoning
        self.stats = PolicyStats()

        self.faction_id = 0
        self._system: str | None = None
        self._plan: list[dict] = []
        self._planned_turn = -1

    def reset(self, faction_id: int, seed: int) -> None:
        self.faction_id = faction_id
        self._plan = []
        self._planned_turn = -1
        # Le message système ne dépend que des règles : construit une fois, il reste
        # identique sur toute la partie et devient cachable côté fournisseur.
        self._system = build_system(self.cfg, self.catalog)

    # ── décision ────────────────────────────────────────────────────────────

    def act(self, observation: dict, info: dict) -> int | dict:
        view = info["view"]
        index = info["index"]

        if view["turn"] != self._planned_turn:
            self._plan = self._request_plan(view, index)
            self._planned_turn = view["turn"]

        legal = {item["index"] for item in view["legal_actions"]}
        while self._plan:
            move = self._plan.pop(0)
            if move["index"] in legal:
                return action(move["index"], move.get("argument", ""))
            # Coup devenu injouable (budget dépassé, cible disparue) : on le compte
            # et on passe au suivant, sans jamais lui substituer un coup au hasard.
            self.stats.rejected_moves += 1
            logger.info(
                "[%s F%s T%s] coup %s hors du masque au moment de jouer — ignoré",
                self.name, self.faction_id, view["turn"], move["index"],
            )

        return action(index.end_turn_index)

    # ── planification ───────────────────────────────────────────────────────

    def _request_plan(self, view: dict, index) -> list[dict]:
        """Un appel, plus des relances ciblées si la sortie est inexploitable."""
        self.stats.turns += 1
        legal = {item["index"] for item in view["legal_actions"]}
        note: str | None = None

        for attempt in range(int(self.cfg.llm.plan_retries) + 1):
            payload = self.backend.complete(
                system=self._system,
                user=build_user(self.cfg, self.catalog, view, retry_note=note),
                tool=PLAN_TOOL,
            )
            if payload is None:
                note = None                            # panne réseau : inutile de recadrer
                continue

            plan, problem = self._validate(payload, legal, index)
            if problem is None:
                self._record(view, payload, plan)
                if not plan:
                    self.stats.empty_plans += 1
                self.stats.planned_moves += len(plan)
                return plan

            note = problem
            self.stats.retries += 1
            logger.info(
                "[%s F%s T%s] plan invalide (essai %s) : %s",
                self.name, self.faction_id, view["turn"], attempt + 1, problem,
            )

        # Relances épuisées : no-op assumé, et signalé. Un tour muet n'est PAS un
        # choix stratégique et ne doit jamais être crédité au modèle.
        self.stats.mute_turns += 1
        logger.warning(
            "[%s F%s T%s] aucun plan exploitable — tour perdu",
            self.name, self.faction_id, view["turn"],
        )
        self._record(view, {"reasoning": ""}, [], status="mute")
        return []

    def _validate(self, payload: dict, legal: set[int], index) -> tuple[list[dict], str | None]:
        """Valide la forme et la légalité. Renvoie `(plan, message d'erreur)`.

        Le message est destiné au modèle : il doit dire précisément ce qui cloche et
        ce qu'il faut faire, sinon la relance ne sert à rien.
        """
        raw = payload.get("plan")
        if raw is None:
            return [], "champ 'plan' absent. Renvoie une liste de coups, éventuellement vide."
        if not isinstance(raw, list):
            return [], f"'plan' doit être une liste, reçu {type(raw).__name__}."

        plan: list[dict] = []
        unknown: list = []
        for item in raw:
            if isinstance(item, int):
                item = {"index": item}
            if not isinstance(item, dict) or "index" not in item:
                return [], "chaque coup doit être un objet avec un champ entier 'index'."
            try:
                move_index = int(item["index"])
            except (TypeError, ValueError):
                return [], f"index non entier : {item['index']!r}."
            if move_index not in legal:
                unknown.append(move_index)
                continue
            plan.append({
                "index": move_index,
                "argument": str(item.get("argument", "") or ""),
            })

        if unknown and not plan:
            sample = sorted(legal)[:12]
            return [], (
                f"aucun index valide : {unknown} n'existe(nt) pas dans la liste des "
                f"coups légaux. Choisis parmi les index proposés, par exemple {sample}."
            )
        if unknown:
            # Plan partiellement bon : on le garde et on compte les rejets, plutôt
            # que de relancer et de payer un appel de plus.
            self.stats.rejected_moves += len(unknown)
        return plan, None

    def _record(self, view: dict, payload: dict, plan: list[dict], status: str = "ok") -> None:
        if not self.keep_reasoning:
            return
        self.stats.reasonings.append({
            "turn": view["turn"],
            "faction_id": self.faction_id,
            "status": status,
            "reasoning": str(payload.get("reasoning", "") or "")[:2000],
            "plan": [move["index"] for move in plan],
        })

    # ── restitution ─────────────────────────────────────────────────────────

    def report(self) -> dict:
        return {
            "model": self.name,
            "policy": self.stats.to_dict(),
            "backend": self.backend.stats.to_dict(),
            # Le raisonnement est la matière première du finding final : sans lui,
            # on sait QUI gagne mais pas POURQUOI un modèle joue comme il joue.
            "reasonings": self.stats.reasonings,
        }
