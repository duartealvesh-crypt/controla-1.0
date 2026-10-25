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

Mémoire inter-tours
-------------------
Sans mémoire, chaque appel ne voit que le règlement et l'état du tour : le modèle
ignore ce qu'il a planifié, promis ou écrit la veille. Avec `memory=True`, les
échanges des tours précédents — le message reçu et la réponse rendue — sont
rejoués comme une conversation entre le système et le tour courant.

C'est un **paramètre d'étude**, pas une amélioration par défaut : le même modèle
peut occuper deux sièges, avec et sans mémoire, et le classement dit ce que la
mémoire lui rapporte (cf. `models.yaml`, variantes `@memoire`).

Deux choix de conception, faits pour que l'étude mesure la mémoire et rien d'autre :

* **La conversation ne fait que s'allonger.** Le préfixe système + historique est
  donc identique d'un appel à l'autre, et un fournisseur à cache de prompt le
  ressert au lieu de le refacturer. Une fenêtre glissante, ou un résumé réécrit
  à chaque tour, casserait ce préfixe et ferait payer la mémoire plein tarif.
* **La réponse mémorisée est celle que le modèle a rendue** — son raisonnement
  explicite et son plan — et non son raisonnement NATIF. Ce dernier n'existe que
  chez les modèles à réflexion : le rejouer donnerait aux uns une mémoire plus
  riche qu'aux autres, et l'on mesurerait l'architecture plutôt que la mémoire.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

from controla.policies.base import Policy, action
from controla.policies.llm.backend import CallStats, LLMBackend
from controla.policies.llm.prompt import PLAN_TOOL, build_system, build_user

logger = logging.getLogger("controla.llm.policy")

# Réponse mémorisée pour un tour où le modèle n'a rien rendu d'exploitable. La
# conversation doit alterner utilisateur / assistant : sauter le tour décalerait
# tout l'historique, et le taire ferait croire au modèle qu'il avait joué.
MUTE_MEMORY = "[aucun plan exploitable rendu ce tour-là — tour perdu]"

# Une réponse vide — aucun coup ET aucun raisonnement — n'est pas un choix de
# passer son tour : mesuré sur dots-3-note, son raisonnement interne concluait
# par un plan de quatre coups, puis le tool call arrivait vide. ~15 % des tours
# étaient ainsi perdus en étant comptés comme des décisions. On la relance une
# fois, avec cette consigne ; un plan vide ACCOMPAGNÉ d'un raisonnement reste,
# lui, un choix légitime.
BLANK_ANSWER_NOTE = (
    "ta réponse était vide : aucun coup et aucun raisonnement. Si tu choisis "
    "de ne rien jouer, explique pourquoi dans 'reasoning' ; sinon, donne ton plan."
)
BLANK_RETRIES = 1

# Un raisonnement compte s'il contient au moins une vraie phrase : ce seuil de
# mots écarte les fragments de sortie cassée observés en partie réelle — `": "`,
# `":dots_ellipsis"`, `"}]"` — qu'un simple test « non vide » laissait passer.
MIN_REASONING_WORDS = 3
_WORD = re.compile(r"[^\W\d_]{2,}")


def has_reasoning(text) -> bool:
    """Le texte contient-il au moins `MIN_REASONING_WORDS` mots de deux lettres ?

    Volontairement grossier : on ne juge pas la qualité d'un raisonnement, on
    distingue une explication — même brève — d'un débris de sérialisation.
    """
    return len(_WORD.findall(str(text or ""))) >= MIN_REASONING_WORDS


@dataclass
class PolicyStats:
    """Ce qu'il faut savoir pour ne pas confondre un modèle prudent et un modèle muet."""

    turns: int = 0
    planned_moves: int = 0
    rejected_moves: int = 0        # index hors de la liste légale au moment de jouer
    empty_plans: int = 0           # le modèle a répondu, mais n'a rien joué
    mute_turns: int = 0            # aucune réponse exploitable après relances
    # Réponses vides (ni coup ni raisonnement), relancées ou non, et sous-ensemble
    # des tours muets qu'elles ont coûtés une fois la relance épuisée.
    blank_answers: int = 0
    blank_turns: int = 0
    # Sous-ensemble des tours muets : le plafond de sortie a coupé le modèle. Ce
    # n'est pas une défaillance du modèle mais un réglage trop serré — à
    # distinguer, sans quoi on classe un modèle capable comme défaillant.
    truncated_turns: int = 0
    # Sous-ensemble des tours muets : le filtre de contenu du fournisseur a coupé
    # le modèle. Compté à part parce qu'il ne frappe pas au hasard — il tombe
    # quand le modèle raisonne sur une attaque — et biaise donc le classement.
    filtered_turns: int = 0
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
            "blank_answers": self.blank_answers,
            "blank_turns": self.blank_turns,
            "truncated_turns": self.truncated_turns,
            "filtered_turns": self.filtered_turns,
            "retries": self.retries,
        }


class LLMPolicy(Policy):
    """Policy adossée à un `LLMBackend`."""

    def __init__(self, backend: LLMBackend, cfg, catalog, keep_reasoning: bool = True,
                 memory: bool = False):
        self.backend = backend
        # Le nom du SIÈGE, pas forcément l'identifiant API : deux variantes d'un
        # même modèle doivent apparaître comme deux participants distincts.
        self.name = getattr(backend, "label", None) or backend.model
        self.cfg = cfg
        self.catalog = catalog
        self.keep_reasoning = keep_reasoning
        self.memory = bool(memory)
        self.stats = PolicyStats()
        # Ce que CE siège a consommé dans CETTE partie. Le backend, lui, est
        # partagé sur la campagne : ses compteurs cumulés, additionnés partie
        # après partie, surcomptaient chaque appel autant de fois qu'il restait
        # de parties à jouer.
        self.call_stats = CallStats()

        self.faction_id = 0
        self._system: str | None = None
        self._plan: list[dict] = []
        self._planned_turn = -1
        self._history: list[dict] = []

    def reset(self, faction_id: int, seed: int) -> None:
        self.faction_id = faction_id
        self._plan = []
        self._planned_turn = -1
        # Une partie neuve est une conversation neuve : la mémoire ne traverse
        # jamais deux parties, sans quoi la seconde serait jouée avec la
        # connaissance de la première.
        self._history = []
        # Le message système ne dépend que des règles : construit une fois, il reste
        # identique sur toute la partie et devient cachable côté fournisseur.
        self._system = build_system(self.cfg, self.catalog, memory=self.memory)

    # ── décision ────────────────────────────────────────────────────────────

    def plan_ahead(self, info: dict) -> None:
        """Planifie le tour AVANT que ce siège ne reçoive la main.

        L'arène l'appelle pour les quatre dieux en parallèle, en début de ronde.
        C'est exact, pas approché : rien n'est appliqué avant la barrière, et
        les coups soumis par un dieu ne modifient que SES points — la vue d'un
        rival au début de la ronde est donc celle qu'il aurait reçue à son tour
        de parole. Sans cela, chaque ronde attendait quatre appels l'un après
        l'autre.

        Idempotent : un tour déjà planifié n'est pas redemandé.
        """
        view = info["view"]
        if view["turn"] != self._planned_turn:
            self._plan = self._request_plan(view, info["index"])
            self._planned_turn = view["turn"]

    def act(self, observation: dict, info: dict) -> int | dict:
        view = info["view"]
        index = info["index"]

        # Planifié d'avance si l'arène l'a demandé ; sinon, maintenant.
        self.plan_ahead(info)

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
        user = ""
        attempts = 0
        blanks = 0
        # Durée MURALE du tour, relances comprises : c'est ce que le joueur — ou
        # la ronde — attend réellement, pas la latence d'un appel isolé.
        self._turn_started = time.perf_counter()

        for attempt in range(int(self.cfg.llm.plan_retries) + 1):
            attempts = attempt + 1
            user = build_user(self.cfg, self.catalog, view, retry_note=note)
            payload = self.backend.complete(
                system=self._system,
                user=user,
                tool=PLAN_TOOL,
                history=self._history if self.memory else (),
                sink=self.call_stats,
            )
            outcome = getattr(self.backend, "last_call", None)

            if payload is None:
                kind = getattr(outcome, "error_kind", None)
                if kind in ("truncated", "filtered", "fatal"):
                    # Relancer à l'identique ne sert à rien : un appel tronqué
                    # le sera encore (même prompt, même plafond), un appel
                    # filtré aussi (mesuré : 6 relances, 6 filtrages), une clé
                    # invalide le restera. On s'arrête, et on le dit.
                    if kind == "filtered":
                        self.stats.filtered_turns += 1
                        logger.warning(
                            "[%s F%s T%s] coupé par le filtre de contenu du "
                            "fournisseur — tour perdu",
                            self.name, self.faction_id, view["turn"],
                        )
                    if kind == "truncated":
                        self.stats.truncated_turns += 1
                        logger.warning(
                            "[%s F%s T%s] sortie tronquée par max_tokens — relever "
                            "`max_tokens` pour ce modèle (models.yaml)",
                            self.name, self.faction_id, view["turn"],
                        )
                    break
                note = None                            # panne réseau : inutile de recadrer
                continue

            plan, problem = self._validate(payload, legal, index)
            if problem is None and not plan and not has_reasoning(payload.get("reasoning")):
                self.stats.blank_answers += 1
                blanks += 1
                if blanks > BLANK_RETRIES:
                    break                              # deuxième réponse vide : on s'arrête
                problem = BLANK_ANSWER_NOTE
            if problem is None:
                self._record(view, payload, plan, outcome=outcome, attempts=attempts)
                self._remember(user, self._memory_entry(payload, plan))
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
        outcome = getattr(self.backend, "last_call", None)
        if blanks > BLANK_RETRIES:
            self.stats.blank_turns += 1
        status = (
            "blank" if blanks > BLANK_RETRIES
            else "truncated" if getattr(outcome, "truncated", False)
            else "filtered" if getattr(outcome, "filtered", False)
            else "mute"
        )
        self._record(view, {"reasoning": ""}, [], status=status,
                     outcome=outcome, attempts=attempts)
        self._remember(user, MUTE_MEMORY)
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

    # ── mémoire ─────────────────────────────────────────────────────────────

    @staticmethod
    def _memory_entry(payload: dict, plan: list[dict]) -> str:
        """La réponse du modèle telle qu'elle sera rejouée au tour suivant.

        Sérialisation DÉTERMINISTE (`sort_keys`) : le cache de prompt est une
        correspondance d'octets, un dict réordonné d'un appel à l'autre suffirait
        à l'invalider pour tout ce qui suit.
        """
        moves = [
            {"index": move["index"], **({"argument": move["argument"]} if move["argument"] else {})}
            for move in plan
        ]
        return json.dumps(
            {"reasoning": str(payload.get("reasoning", "") or ""), "plan": moves},
            ensure_ascii=False, sort_keys=True,
        )

    def _remember(self, user: str, answer: str) -> None:
        """Ajoute l'échange du tour à la conversation — par ajout seulement.

        Le message utilisateur mémorisé est celui qui a VRAIMENT été envoyé, note
        de correction comprise : rejouer un texte différent casserait le préfixe
        mis en cache, et mentirait au modèle sur ce qu'il avait lu.
        """
        if not self.memory or not user:
            return
        self._history.append({"role": "user", "content": user})
        self._history.append({"role": "assistant", "content": answer})

    # ── restitution ─────────────────────────────────────────────────────────

    def _clip(self, text: str | None) -> str | None:
        """Tronque un raisonnement seulement si la config le demande.

        Le défaut est de tout garder : c'est la matière première du finding, et
        la couper au hasard d'un plafond revenait à jeter précisément la fin —
        là où le modèle conclut.
        """
        if text is None:
            return None
        try:
            limit = self.cfg.llm.reasoning_max_chars
        except AttributeError:
            limit = None
        if limit is None:
            return text
        return text[: int(limit)]

    def _record(self, view: dict, payload: dict, plan: list[dict], status: str = "ok",
                outcome=None, attempts: int = 1) -> None:
        if not self.keep_reasoning:
            return
        self.stats.reasonings.append({
            "turn": view["turn"],
            "faction_id": self.faction_id,
            "status": status,
            "reasoning": self._clip(str(payload.get("reasoning", "") or "")),
            # Le raisonnement interne d'un modèle à réflexion, quand le
            # fournisseur l'expose. `None` ne veut pas dire « n'a pas réfléchi »
            # mais « rien n'a été rendu » — beaucoup de modèles le cachent.
            "native_reasoning": self._clip(getattr(outcome, "native_reasoning", None)),
            "finish_reason": getattr(outcome, "finish_reason", None),
            "attempts": attempts,
            "seconds": round(time.perf_counter() - getattr(self, "_turn_started", time.perf_counter()), 3),
            "plan": [move["index"] for move in plan],
        })

    def report(self) -> dict:
        return {
            "model": self.name,
            "api_model": getattr(self.backend, "model", self.name),
            "memory": self.memory,
            "memory_messages": len(self._history),
            "policy": self.stats.to_dict(),
            # Par siège et par partie — c'est ce qui s'additionne correctement
            # sur une campagne. Le cumul du backend reste lisible sur l'objet.
            "backend": self.call_stats.to_dict(),
            # Le raisonnement est la matière première du finding final : sans lui,
            # on sait QUI gagne mais pas POURQUOI un modèle joue comme il joue.
            "reasonings": self.stats.reasonings,
        }
