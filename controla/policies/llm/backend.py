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
    passerait pour un modèle prudent."""

    calls: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)

    def record(self, seconds: float, error_kind: str | None = None) -> None:
        self.calls += 1
        self.latencies.append(seconds)
        if error_kind:
            self.failures[error_kind] = self.failures.get(error_kind, 0) + 1

    @property
    def response_rate(self) -> float:
        return 1.0 if not self.calls else 1.0 - sum(self.failures.values()) / self.calls

    def to_dict(self) -> dict:
        ordered = sorted(self.latencies)
        return {
            "calls": self.calls,
            "failures": dict(self.failures),
            "response_rate": round(self.response_rate, 4),
            "latency_mean": round(sum(ordered) / len(ordered), 3) if ordered else None,
            "latency_max": round(ordered[-1], 3) if ordered else None,
        }


class LLMBackend:
    """Un modèle joignable. Renvoie un dict parsé, ou `None` en cas d'échec."""

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
    ):
        if not _AVAILABLE:                            # pragma: no cover
            raise ImportError(
                "le paquet 'openai' est requis pour les policies LLM : pip install openai"
            )
        self.model = model
        self.tool_support = tool_support
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.stats = CallStats()
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries,
        )

    def _sampling(self) -> dict:
        """N'envoie que les réglages définis : ne jamais forcer un paramètre qu'un
        fournisseur donné rejetterait."""
        return {
            key: value
            for key, value in (
                ("temperature", self.temperature),
                ("max_tokens", self.max_tokens),
                ("seed", self.seed),
            )
            if value is not None
        }

    def complete(self, system: str, user: str, tool: dict) -> dict | None:
        """Un appel. `tool` est la définition de fonction au format OpenAI."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        started = time.perf_counter()
        try:
            payload = (
                self._call_with_tool(messages, tool) if self.tool_support
                else self._call_with_json(messages, tool)
            )
        except Exception as error:                    # noqa: BLE001 — on classe puis on log
            kind = classify(error)
            self.stats.record(time.perf_counter() - started, kind)
            if kind == "fatal":
                logger.error(
                    "[%s] erreur FATALE (clé, nom de modèle ou accès) — ce modèle "
                    "ne jouera jamais : %s",
                    self.model, error,
                )
            else:
                logger.warning("[%s] erreur %s : %s", self.model, kind, error)
            return None

        elapsed = time.perf_counter() - started
        if payload is None:
            self.stats.record(elapsed, "empty")
            logger.warning("[%s] réponse vide ou non parsable", self.model)
            return None
        self.stats.record(elapsed)
        return payload

    # ── modes ───────────────────────────────────────────────────────────────

    def _call_with_tool(self, messages: list[dict], tool: dict) -> dict | None:
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
            return None
        return json.loads(calls[0].function.arguments)

    def _call_with_json(self, messages: list[dict], tool: dict) -> dict | None:
        schema = tool["function"].get("parameters", {})
        messages = list(messages)
        messages[-1] = {
            **messages[-1],
            "content": messages[-1]["content"] + (
                "\n\nRéponds UNIQUEMENT par un objet JSON valide conforme à ce "
                f"schéma, sans texte autour :\n{json.dumps(schema, ensure_ascii=False)}"
            ),
        }
        response = self._client.chat.completions.create(
            model=self.model, messages=messages, **self._sampling(),
        )
        content = (response.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content).strip()
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Dernier recours : isoler le premier objet JSON du texte.
            match = re.search(r"\{.*\}", content, re.S)
            if not match:
                return None
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
