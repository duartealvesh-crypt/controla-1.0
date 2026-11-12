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


def _usage(prompt=0, completion=0, cached=None, reasoning=None):
    details = None if cached is None else SimpleNamespace(cached_tokens=cached)
    completion_details = (
        None if reasoning is None else SimpleNamespace(reasoning_tokens=reasoning)
    )
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        prompt_tokens_details=details,
        completion_tokens_details=completion_details,
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


def test_reasoning_tokens_are_counted_as_a_share_of_the_completion():
    """Chez un modèle à raisonnement, la réflexion EST la sortie (95.6 % mesurés
    sur nemotron-3-nano-omni). Sans ce compteur, un `completion_tokens` élevé ne
    dit pas si le modèle est verbeux ou s'il réfléchit.

    Ventilation et non total : les tokens de raisonnement sont DÉJÀ dans
    `completion_tokens`, les additionner les compterait deux fois.
    """
    backend = _backend([
        _tool_response('{"plan": []}', _usage(prompt=6000, completion=2677, reasoning=2644)),
    ])
    backend.complete("sys", "user", TOOL)

    stats = backend.stats.to_dict()
    assert stats["completion_tokens"] == 2677
    assert stats["reasoning_tokens"] == 2644


def test_a_model_without_reasoning_reports_zero_not_a_crash():
    """`completion_tokens_details` est absent chez la plupart des fournisseurs."""
    backend = _backend([_tool_response('{"plan": []}', _usage(prompt=10, completion=20))])
    backend.complete("sys", "user", TOOL)
    assert backend.stats.reasoning_tokens == 0


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


# ── Ce qui se perdait : raisonnement natif, raison de fin, historique ───────

def _choice_response(message, finish_reason, usage=None):
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_native_reasoning_is_captured_under_each_known_field():
    """Chaque passerelle nomme le champ à sa façon ; aucun ne doit se perdre."""
    def tool_message(**extra):
        call = SimpleNamespace(function=SimpleNamespace(arguments='{"plan": []}'))
        return SimpleNamespace(tool_calls=[call], content=None, **extra)

    cases = [
        {"reasoning": "je pèse mes options"},
        {"reasoning_content": "je pèse mes options"},
        {"reasoning_details": [{"type": "reasoning.text", "text": "je pèse mes options"}]},
    ]
    for extra in cases:
        backend = _backend([_choice_response(tool_message(**extra), "tool_calls")])
        backend.complete("s", "u", TOOL)
        assert backend.last_call.native_reasoning == "je pèse mes options", extra


def test_a_truncated_answer_is_classified_as_truncated_not_empty():
    """Coupé par `max_tokens` : un réglage trop serré, pas un modèle défaillant."""
    message = SimpleNamespace(tool_calls=None, content=None, reasoning="je réfléchis enc")
    backend = _backend([_choice_response(message, "length", _usage(prompt=10, completion=8192))])
    assert backend.complete("s", "u", TOOL) is None
    assert backend.last_call.truncated
    assert backend.last_call.error_kind == "truncated"
    assert backend.stats.failures == {"truncated": 1}
    assert backend.stats.finish_reasons == {"length": 1}
    # Même coupé, le raisonnement partiel est gardé : c'est souvent tout ce qu'il y a.
    assert backend.last_call.native_reasoning == "je réfléchis enc"


def test_cache_writes_are_counted_apart_from_cache_reads():
    usage = _usage(prompt=5000, completion=10)
    usage.prompt_tokens_details = SimpleNamespace(cached_tokens=0, cache_write_tokens=4800)
    stats = CallStats()
    stats.record_usage(usage)
    assert stats.cache_write_tokens == 4800
    assert stats.cached_tokens == 0


def test_history_sits_between_the_system_and_the_turn():
    """Système puis historique : le préfixe qui ne fait que s'allonger."""
    history = [{"role": "user", "content": "tour 0"},
               {"role": "assistant", "content": '{"plan": []}'}]
    backend = _backend([_tool_response('{"plan": []}')])
    backend.complete("règles", "tour 1", TOOL, history=history)
    sent = backend._client.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[0]["content"] == "règles"
    assert sent[-1]["content"] == "tour 1"


def test_cache_control_is_sent_only_when_declared():
    plain = _backend([_tool_response('{"plan": []}')])
    plain.complete("s", "u", TOOL)
    assert "extra_body" not in plain._client.calls[0]

    cached = _backend([_tool_response('{"plan": []}')], cache_control=True)
    cached.complete("s", "u", TOOL)
    assert cached._client.calls[0]["extra_body"] == {"cache_control": {"type": "ephemeral"}}


def test_json_mode_puts_the_schema_in_the_system_not_in_the_turn():
    """Le schéma est constant : dans le système il se cache, et le message du
    tour mémorisé reste exactement celui qui a été envoyé."""
    backend = _backend([_text_response('{"plan": []}')], tool_support=False)
    backend.complete("règles", "tour 1", TOOL)
    sent = backend._client.calls[0]["messages"]
    assert "schéma" in sent[0]["content"]
    assert sent[-1]["content"] == "tour 1"


def test_the_seat_label_and_the_api_model_are_distinct():
    backend = LLMBackend(base_url="http://local", api_key="x",
                         model="openai/gpt-5-nano", label="openai/gpt-5-nano@memoire")
    backend._client = FakeClient([_tool_response('{"plan": []}')])
    backend.complete("s", "u", TOOL)
    assert backend._client.calls[0]["model"] == "openai/gpt-5-nano"
    assert backend.label == "openai/gpt-5-nano@memoire"


def test_each_seat_counts_only_its_own_calls_on_a_shared_backend():
    """Le backend est partagé sur la campagne : ses compteurs sont cumulés.
    Additionner ce cumul partie après partie surcomptait chaque appel ; chaque
    siège tient donc les siens."""
    backend = _backend([
        _tool_response('{"plan": []}', _usage(prompt=100)),
        _tool_response('{"plan": []}', _usage(prompt=100)),
        _tool_response('{"plan": []}', _usage(prompt=100)),
    ])
    first_game, second_game = CallStats(), CallStats()
    backend.complete("s", "u", TOOL, sink=first_game)
    backend.complete("s", "u", TOOL, sink=second_game)
    backend.complete("s", "u", TOOL, sink=second_game)

    assert first_game.calls == 1 and first_game.prompt_tokens == 100
    assert second_game.calls == 2 and second_game.prompt_tokens == 200
    assert backend.stats.calls == 3                  # le cumul reste disponible
    # Ce qui s'additionne sur la campagne vaut bien le total réel.
    assert first_game.calls + second_game.calls == backend.stats.calls


def test_a_content_filter_cut_is_classified_as_filtered():
    """Le filtre du fournisseur n'est ni une réponse vide, ni une troncature :
    il tombe quand le modèle raisonne sur une attaque, et biaise le banc."""
    message = SimpleNamespace(tool_calls=None, content=None, reasoning="je vais attaquer Ambre pour")
    backend = _backend([_choice_response(message, "content_filter")])
    assert backend.complete("s", "u", TOOL) is None
    assert backend.last_call.filtered and not backend.last_call.truncated
    assert backend.last_call.error_kind == "filtered"
    assert backend.stats.failures == {"filtered": 1}
    assert backend.last_call.native_reasoning == "je vais attaquer Ambre pour"


def test_reasoning_effort_is_sent_only_when_declared():
    plain = _backend([_tool_response('{"plan": []}')])
    plain.complete("s", "u", TOOL)
    assert "extra_body" not in plain._client.calls[0]

    low = _backend([_tool_response('{"plan": []}')], reasoning_effort="low", cache_control=True)
    low.complete("s", "u", TOOL)
    assert low._client.calls[0]["extra_body"] == {
        "reasoning": {"effort": "low"}, "cache_control": {"type": "ephemeral"},
    }


# ── tool_choice : forcé ou libre ────────────────────────────────────────────

def test_tool_choice_is_forced_by_default_and_auto_on_request():
    forced = _backend([_tool_response('{"plan": []}')])
    forced.complete("s", "u", TOOL)
    assert forced._client.calls[0]["tool_choice"]["function"]["name"] == "plan_turn"

    free = _backend([_tool_response('{"plan": []}')], tool_choice="auto")
    free.complete("s", "u", TOOL)
    assert free._client.calls[0]["tool_choice"] == "auto"


def test_auto_mode_reads_a_plan_written_as_json_text():
    """Un plan rendu dans le texte reste un plan : pas de relance payée pour ça."""
    backend = _backend([_text_response('```json\n{"reasoning": "r", "plan": [3]}\n```')],
                       tool_choice="auto")
    assert backend.complete("s", "u", TOOL) == {"reasoning": "r", "plan": [3]}


def test_auto_mode_classifies_prose_without_tool_call():
    backend = _backend([_text_response("Je pense attaquer Ambre ce tour.")], tool_choice="auto")
    assert backend.complete("s", "u", TOOL) is None
    assert backend.last_call.error_kind == "no_tool"
