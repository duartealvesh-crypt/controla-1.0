"""Le backend LLM : comptage des tokens, classification, robustesse du parsing.

Les tests de `test_llm_policy.py` simulent au niveau de `complete()` — c'est le
bon niveau pour éprouver la *policy*, mais il court-circuite tout ce que fait le
backend. Or ce qu'il fait porte deux décisions du banc :

* **le budget**, qui est LA contrainte de campagne (FINDINGS n° 5) et n'est pas
  déductible du nombre d'appels ;
* **le cache**, dont on ne peut pas se contenter de supposer qu'il est actif.

D'où un faux client OpenAI ici, plutôt qu'un faux backend.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from controla.policies.llm.backend import CallStats, LLMBackend

TOOL = {
    "type": "function",
    "function": {"name": "plan_turn", "parameters": {"type": "object"}},
}


def _usage(prompt=0, completion=0, cached=None):
    details = None if cached is None else SimpleNamespace(cached_tokens=cached)
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        prompt_tokens_details=details,
    )


def _tool_response(arguments: str, usage=None):
    call = SimpleNamespace(function=SimpleNamespace(arguments=arguments))
    message = SimpleNamespace(tool_calls=[call], content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _text_response(content: str, usage=None):
    message = SimpleNamespace(tool_calls=None, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeClient:
    """Le minimum du client OpenAI qu'utilise le backend."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _backend(responses, **kwargs):
    backend = LLMBackend(
        base_url="http://local", api_key="x", model="fake", **kwargs,
    )
    backend._client = FakeClient(responses)
    return backend


# ── Comptage des tokens ─────────────────────────────────────────────────────

def test_tokens_accumulate_across_calls():
    backend = _backend([
        _tool_response('{"plan": []}', _usage(prompt=2400, completion=120)),
        _tool_response('{"plan": []}', _usage(prompt=2600, completion=90)),
    ])
    backend.complete("sys", "user", TOOL)
    backend.complete("sys", "user", TOOL)

    stats = backend.stats.to_dict()
    assert stats["prompt_tokens"] == 5000
    assert stats["completion_tokens"] == 210
    assert stats["calls"] == 2


def test_an_unusable_answer_still_costs_its_tokens():
    """Un modèle qui répond à côté a consommé le budget. L'oublier ferait
    passer le plus coûteux du panel pour le plus économe."""
    backend = _backend([_text_response("", _usage(prompt=2400, completion=800))],
                       tool_support=False)
    assert backend.complete("sys", "user", TOOL) is None
    assert backend.stats.prompt_tokens == 2400
    assert backend.stats.completion_tokens == 800
    assert backend.stats.failures.get("empty") == 1


def test_cache_hit_rate_is_none_until_something_is_measured():
    """Zéro afficherait « cache inactif » là où l'on ne sait rien."""
    stats = CallStats()
    assert stats.cache_hit_rate is None
    stats.record_usage(_usage(prompt=1000, completion=10, cached=800))
    assert stats.cache_hit_rate == 0.8


def test_missing_usage_block_costs_a_counter_not_a_campaign():
    """`usage` est optionnel dans la spec OpenAI et des passerelles l'omettent."""
    backend = _backend([_tool_response('{"plan": []}', None)])
    assert backend.complete("sys", "user", TOOL) == {"plan": []}
    assert backend.stats.prompt_tokens == 0
    assert backend.stats.cache_hit_rate is None


# ── Robustesse du parsing ───────────────────────────────────────────────────

def test_malformed_tool_arguments_are_an_empty_answer_not_a_crash():
    """Une panne de FORME du modèle doit se ranger avec les réponses vides, pas
    faire tomber la campagne."""
    backend = _backend([_tool_response("{ pas du json", _usage(prompt=10))])
    assert backend.complete("sys", "user", TOOL) is None
    assert backend.stats.failures.get("empty") == 1


def test_json_mode_strips_fences_and_recovers_a_wrapped_object():
    backend = _backend([
        _text_response('```json\n{"plan": [{"index": 3}]}\n```', _usage(prompt=5)),
        _text_response('Voici mon plan : {"plan": []} — voilà.', _usage(prompt=5)),
    ], tool_support=False)
    assert backend.complete("s", "u", TOOL) == {"plan": [{"index": 3}]}
    assert backend.complete("s", "u", TOOL) == {"plan": []}


# ── Réglages d'échantillonnage ──────────────────────────────────────────────

def test_only_defined_sampling_settings_are_sent():
    """`None` ne doit RIEN envoyer : c'est ce qui permet à un modèle refusant
    `temperature` de tourner dans le même panel que les autres."""
    backend = _backend([_tool_response('{"plan": []}')],
                       temperature=None, seed=None, max_tokens=1024)
    backend.complete("s", "u", TOOL)
    sent = backend._client.calls[0]
    assert "temperature" not in sent
    assert "seed" not in sent
    assert sent["max_tokens"] == 1024


def test_defined_sampling_settings_are_sent():
    backend = _backend([_tool_response('{"plan": []}')], temperature=0.3, seed=7)
    backend.complete("s", "u", TOOL)
    sent = backend._client.calls[0]
    assert sent["temperature"] == 0.3
    assert sent["seed"] == 7


def test_tool_call_forces_the_expected_function():
    backend = _backend([_tool_response('{"plan": []}')])
    backend.complete("s", "u", TOOL)
    sent = backend._client.calls[0]
    assert sent["tool_choice"]["function"]["name"] == "plan_turn"
    assert json.dumps(sent["tools"])                 # sérialisable, donc envoyable
