"""Client LLM compatible OpenAI, avec deux modes de sortie structurée.

Changer de fournisseur est un changement de deux valeurs (`base_url`, `api_key`) :
OpenAI, Anthropic, Groq, NVIDIA NIM et OpenRouter exposent tous une API compatible.

Deux modes, déclarés par modèle :

* `tool_support=True`  — *tool calling* forcé. Le fournisseur garantit la forme.
* `tool_support=False` — le schéma JSON est annexé au message et la réponse est
  parsée. Moins fiable, mais indispensable pour les modèles sans tool calling.

**Pourquoi classer les erreurs.** Un modèle muet joue « pass » et disparaît du
benchmark sans bruit. Une erreur d'authentification veut dire qu'il n'a JAMAIS joué
(le classement est alors une illusion) ; un rate-limit est un incident passager déjà
retenté par le SDK. Confondre les deux, c'est publier un classement faux.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger("controla.llm")

try:
    from openai import OpenAI
    _AVAILABLE = True
except ImportError:                                   # pragma: no cover
    _AVAILABLE = False

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        NotFoundError,
        PermissionDeniedError,
        RateLimitError,
    )
    # FATALES : mauvaise clé, mauvais nom de modèle, accès refusé. Le modèle ne
    # jouera JAMAIS tant qu'on n'aura pas corrigé la configuration.
    _FATAL_ERRORS = (AuthenticationError, PermissionDeniedError, NotFoundError)
    # REQUÊTE REFUSÉE : ce message-ci a déplu (trop long, champ inattendu…). Le
    # modèle peut très bien répondre au suivant — on l'a mesuré. Confondre ce cas
    # avec une clé invalide ferait crier au loup sur une campagne qui va bien.
    _BAD_REQUEST_ERRORS = (BadRequestError,)
    # TRANSITOIRES : déjà retentées `max_retries` fois par le SDK.
    _TRANSIENT_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)
except ImportError:                                   # pragma: no cover
    _FATAL_ERRORS = ()
    _BAD_REQUEST_ERRORS = ()
    _TRANSIENT_ERRORS = ()


def classify(error: Exception) -> str:
    if _FATAL_ERRORS and isinstance(error, _FATAL_ERRORS):
        return "fatal"
    if _BAD_REQUEST_ERRORS and isinstance(error, _BAD_REQUEST_ERRORS):
        return "bad_request"
    if _TRANSIENT_ERRORS and isinstance(error, _TRANSIENT_ERRORS):
        return "transient"
    return "error"


@dataclass
class CallStats:
    """Santé d'un modèle sur une campagne. Sans ces compteurs, un modèle muet
    passerait pour un modèle prudent.

    Les tokens y sont comptés parce que le budget est LA contrainte de campagne
    (FINDINGS n° 5) et qu'il n'est pas déductible du nombre d'appels : deux
    modèles au même nombre d'appels peuvent différer d'un facteur trois selon la
    longueur de leur raisonnement. `cached_tokens` isole ce que le fournisseur a
    servi depuis son cache — c'est la seule façon de vérifier qu'un cache
    supposé actif l'est vraiment, plutôt que de le supposer.

    `reasoning_tokens` isole de même la part de la sortie passée à réfléchir.
    Mesuré sur `nemotron-3-nano-omni` : **95.6 % des tokens de sortie**. Sans ce
    compteur, la seule chose visible d'un modèle à raisonnement est un
    `completion_tokens` inexpliqué, et l'on ne peut ni distinguer un modèle
    verbeux d'un modèle qui réfléchit, ni chiffrer une campagne.
    """

    calls: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    # Tokens ÉCRITS dans le cache (facturés plus cher que l'entrée normale chez
    # Anthropic). Sans eux, un cache qui s'écrit à chaque appel sans jamais être
    # relu passerait pour un cache neutre, alors qu'il coûte.
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    # Pourquoi chaque génération s'est arrêtée (`stop`, `tool_calls`, `length`…).
    # `length` est le cas qui compte : le plafond de sortie a coupé le modèle,
    # typiquement en plein raisonnement, avant qu'il n'émette son plan.
    finish_reasons: dict[str, int] = field(default_factory=dict)

    def record(self, seconds: float, error_kind: str | None = None) -> None:
        self.calls += 1
        self.latencies.append(seconds)
        if error_kind:
            self.failures[error_kind] = self.failures.get(error_kind, 0) + 1

    def record_usage(self, usage) -> None:
        """Absorbe le bloc `usage` d'une réponse, quel que soit le fournisseur.

        Défensif par construction : `usage` est optionnel dans la spec OpenAI et
        plusieurs passerelles l'omettent ou le nomment autrement. Un champ
        manquant doit coûter un compteur à zéro, jamais une campagne plantée.

        `cached_tokens` et `reasoning_tokens` sont des VENTILATIONS, pas des
        totaux à part : le premier est déjà compris dans `prompt_tokens`, le
        second dans `completion_tokens`. Les additionner reviendrait à compter
        deux fois les mêmes tokens.
        """
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details is not None else 0
        self.cached_tokens += int(cached or 0)
        written = getattr(details, "cache_write_tokens", 0) if details is not None else 0
        self.cache_write_tokens += int(written or 0)
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning = (
            getattr(completion_details, "reasoning_tokens", 0)
            if completion_details is not None else 0
        )
        self.reasoning_tokens += int(reasoning or 0)

    @property
    def response_rate(self) -> float:
        return 1.0 if not self.calls else 1.0 - sum(self.failures.values()) / self.calls

    @property
    def cache_hit_rate(self) -> float | None:
        """Part du prompt servie par le cache du fournisseur.

        `None` tant qu'aucun token n'a été compté — un zéro afficherait « cache
        inactif » là où l'on ne sait simplement rien.
        """
        if not self.prompt_tokens:
            return None
        return self.cached_tokens / self.prompt_tokens

    def to_dict(self) -> dict:
        ordered = sorted(self.latencies)
        return {
            "calls": self.calls,
            "failures": dict(self.failures),
            "response_rate": round(self.response_rate, 4),
            "latency_mean": round(sum(ordered) / len(ordered), 3) if ordered else None,
            "latency_max": round(ordered[-1], 3) if ordered else None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "finish_reasons": dict(self.finish_reasons),
            "cache_hit_rate": (
                None if self.cache_hit_rate is None else round(self.cache_hit_rate, 4)
            ),
        }


@dataclass
class CallOutcome:
    """Ce que le dernier appel a laissé en dehors du plan lui-même.

    Exposé à part plutôt que glissé dans le dict renvoyé par `complete()` : le
    plan est la sortie du MODÈLE, ceci est l'observation du TRANSPORT. Les
    mélanger obligerait chaque lecteur du plan à filtrer des clés techniques.
    """

    finish_reason: str | None = None
    # Le raisonnement NATIF d'un modèle à réflexion (`message.reasoning` chez
    # OpenRouter, `reasoning_content` chez DeepSeek et les serveurs vLLM). Il ne
    # transite pas par le tool call : sans capture explicite, on en paie les
    # tokens et on en jette le texte.
    native_reasoning: str | None = None
    error_kind: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _field(block, key: str):
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def extract_native_reasoning(message) -> str | None:
    """Le raisonnement natif d'un message, sous l'un des noms en usage.

    Il n'y a pas de standard : chaque passerelle a choisi son champ. On les
    essaie dans l'ordre, et `reasoning_details` (liste de blocs typés) en
    dernier recours.
    """
    for name in ("reasoning", "reasoning_content"):
        value = getattr(message, name, None)
        if isinstance(value, str) and value.strip():
            return value
    details = getattr(message, "reasoning_details", None)
    if isinstance(details, list):
        parts = []
        for block in details:
            text = _field(block, "text") or _field(block, "summary")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        if parts:
            return "\n\n".join(parts)
    return None


class LLMBackend:
    """Un modèle joignable. Renvoie un dict parsé, ou `None` en cas d'échec.

    `model` est l'identifiant ENVOYÉ à l'API ; `label` est le nom du SIÈGE, celui
    qui apparaît dans les classements. Ils diffèrent dès qu'un même modèle est
    déclaré en plusieurs variantes (avec et sans mémoire, par exemple) : deux
    sièges, deux classements, un seul modèle appelé.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        tool_support: bool = True,
        timeout: float = 60.0,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        label: str | None = None,
        cache_control: bool = False,
    ):
        if not _AVAILABLE:                            # pragma: no cover
            raise ImportError(
                "le paquet 'openai' est requis pour les policies LLM : pip install openai"
            )
        self.model = model
        self.label = label or model
        self.tool_support = tool_support
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        # Cache de prompt EXPLICITE (Anthropic via OpenRouter). Les fournisseurs
        # à cache implicite (OpenAI, DeepSeek, Gemini 2.5+) n'en ont pas besoin :
        # ils cachent tout préfixe stable au-delà de leur seuil. Le marqueur est
        # posé à la RACINE de la requête, ce qui place le point de cache sur le
        # dernier bloc — exactement ce qu'il faut à une conversation qui grandit
        # par ajouts successifs.
        self.cache_control = cache_control
        self.stats = CallStats()
        self.last_call = CallOutcome()
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries,
            # En-têtes propres au fournisseur (attribution OpenRouter, par
            # exemple). Ils ne changent pas la réponse ; les faire passer par le
            # client plutôt que par chaque appel garantit qu'aucun chemin ne les
            # oublie.
            default_headers=default_headers or None,
        )

    def _sampling(self) -> dict:
        """N'envoie que les réglages définis : ne jamais forcer un paramètre qu'un
        fournisseur donné rejetterait."""
        options = {
            key: value
            for key, value in (
                ("temperature", self.temperature),
                ("max_tokens", self.max_tokens),
                ("seed", self.seed),
            )
            if value is not None
        }
        if self.cache_control:
            options["extra_body"] = {"cache_control": {"type": "ephemeral"}}
        return options

    def complete(
        self, system: str, user: str, tool: dict, history: list[dict] | tuple = (),
        sink: CallStats | None = None,
    ) -> dict | None:
        """Un appel. `tool` est la définition de fonction au format OpenAI.

        `history` est la mémoire de la policy : les échanges des tours
        précédents, insérés ENTRE le système et le message du tour. L'ordre n'est
        pas indifférent — système puis historique forment un préfixe qui ne fait
        que s'allonger d'un tour à l'autre, donc que le cache du fournisseur peut
        resservir à l'identique.

        `sink` reçoit les mêmes compteurs que `self.stats`. Le backend est
        partagé sur toute une campagne — et entre deux sièges s'ils jouent le
        même modèle — : ses compteurs sont donc CUMULÉS. La policy passe ici les
        siens pour savoir ce que coûte SA partie, sans soustraire quoi que ce
        soit après coup.
        """
        sinks = (self.stats,) if sink is None else (self.stats, sink)
        messages = [
            {"role": "system", "content": self._system(system, tool)},
            *history,
            {"role": "user", "content": user},
        ]
        self.last_call = CallOutcome()
        started = time.perf_counter()
        try:
            payload, response = (
                self._call_with_tool(messages, tool) if self.tool_support
                else self._call_with_json(messages)
            )
        except Exception as error:                    # noqa: BLE001 — on classe puis on log
            kind = classify(error)
            self.last_call.error_kind = kind
            for stats in sinks:
                stats.record(time.perf_counter() - started, kind)
            if kind == "fatal":
                logger.error(
                    "[%s] erreur FATALE (clé, nom de modèle ou accès) — ce modèle "
                    "ne jouera jamais : %s",
                    self.label, error,
                )
            else:
                logger.warning("[%s] erreur %s : %s", self.label, kind, error)
            return None

        elapsed = time.perf_counter() - started
        # Les tokens sont comptés même quand la réponse est inexploitable : un
        # modèle qui répond à côté a quand même consommé le budget, et l'oublier
        # ferait paraître économe le modèle le plus coûteux du panel.
        for stats in sinks:
            stats.record_usage(getattr(response, "usage", None))
        choices = getattr(response, "choices", None)
        if choices:
            finish = getattr(choices[0], "finish_reason", None)
            self.last_call.finish_reason = finish
            if finish:
                for stats in sinks:
                    stats.finish_reasons[finish] = stats.finish_reasons.get(finish, 0) + 1
            self.last_call.native_reasoning = extract_native_reasoning(choices[0].message)

        if payload is None:
            # « Tronqué » et « vide » appellent deux remèdes opposés : relancer
            # un appel tronqué à l'identique le tronquera encore, alors qu'une
            # réponse vide peut très bien ne pas se reproduire.
            kind = "truncated" if self.last_call.truncated else "empty"
            self.last_call.error_kind = kind
            for stats in sinks:
                stats.record(elapsed, kind)
            logger.warning(
                "[%s] réponse %s", self.label,
                "tronquée par max_tokens" if kind == "truncated" else "vide ou non parsable",
            )
            return None
        for stats in sinks:
            stats.record(elapsed)
        return payload

    # ── modes ───────────────────────────────────────────────────────────────

    def _system(self, system: str, tool: dict) -> str:
        """Le message système effectivement envoyé.

        En mode JSON, le schéma de sortie y est annexé plutôt qu'au message du
        tour : il est identique à chaque appel, il appartient donc au préfixe
        stable — et en mémoire, le message utilisateur rejoué au tour suivant
        reste exactement celui qui avait été envoyé.
        """
        if self.tool_support:
            return system
        schema = tool["function"].get("parameters", {})
        return system + (
            "\n\nRéponds UNIQUEMENT par un objet JSON valide conforme à ce "
            f"schéma, sans texte autour :\n{json.dumps(schema, ensure_ascii=False)}"
        )

    def _call_with_tool(self, messages: list[dict], tool: dict) -> tuple[dict | None, object]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function",
                         "function": {"name": tool["function"]["name"]}},
            **self._sampling(),
        )
        calls = getattr(response.choices[0].message, "tool_calls", None)
        if not calls:
            return None, response
        try:
            return json.loads(calls[0].function.arguments), response
        except json.JSONDecodeError:
            # Un tool call dont les arguments ne sont pas du JSON valide : rare,
            # mais c'est une panne de FORME du modèle, pas du réseau. La compter
            # comme une réponse vide la range au bon endroit dans les métriques.
            return None, response

    def _call_with_json(self, messages: list[dict]) -> tuple[dict | None, object]:
        response = self._client.chat.completions.create(
            model=self.model, messages=messages, **self._sampling(),
        )
        content = (response.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content).strip()
        if not content:
            return None, response
        try:
            return json.loads(content), response
        except json.JSONDecodeError:
            # Dernier recours : isoler le premier objet JSON du texte.
            match = re.search(r"\{.*\}", content, re.S)
            if not match:
                return None, response
            try:
                return json.loads(match.group()), response
            except json.JSONDecodeError:
                return None, response
