"""La mémoire inter-tours, paramètre d'étude : ce qui est rejoué, et comment.

L'étude oppose un modèle à lui-même, avec et sans mémoire. Elle ne vaut que si
la mémoire est la SEULE différence entre les deux sièges — d'où des tests qui
vérifient ce qui est réellement envoyé, et pas seulement que « ça tourne » :

* sans mémoire, aucun historique ne part, jamais ;
* avec mémoire, l'historique d'un appel est un PRÉFIXE exact de celui du
  suivant — la condition pour que le cache de prompt du fournisseur serve ;
* la mémoire ne traverse jamais deux parties.
"""

from __future__ import annotations

import json
import re

import pytest

from controla.arena.play import MatchSpec, play_match
from controla.policies.archetypes import build_policy
from controla.policies.llm.backend import CallOutcome
from controla.policies.llm.policy import MUTE_MEMORY, LLMPolicy
from controla.policies.llm.prompt import MEMORY_NOTE
from controla.policies.llm.registry import (
    ENTRY_KEYS,
    api_model,
    load_models,
    memory_enabled,
    validate_entry,
)
from controla.registry.catalog import build_catalog
from tests.helpers import base_config


class ScriptedBackend:
    """Joue toujours le premier coup payant listé, et enregistre chaque appel."""

    model = "fake/llm"
    label = "fake/llm"

    def __init__(self, mute_turns: tuple[int, ...] = ()):
        self.calls: list[dict] = []
        self.mute_turns = set(mute_turns)
        self.last_call = CallOutcome()

        class _Stats:
            def to_dict(self):
                return {}

        self.stats = _Stats()

    def complete(self, system, user, tool, history=(), sink=None):
        self.calls.append({
            "system": system, "user": user,
            "history": [dict(message) for message in history],
        })
        turn = int(re.search(r"TOUR (\d+)", user).group(1))
        self.last_call = CallOutcome(finish_reason="tool_calls")
        if turn in self.mute_turns:
            return None
        indices = [int(i) for i in re.findall(r"\[(\d+)\]", user)]
        plan = [{"index": indices[0]}] if indices else []
        return {"reasoning": f"tour {turn}", "plan": plan}


def _play(memory: bool, seed: int = 4, mute_turns=()):
    cfg = base_config()
    catalog = build_catalog(cfg)
    backend = ScriptedBackend(mute_turns)
    policy = LLMPolicy(backend, cfg, catalog, memory=memory)
    result = play_match(
        MatchSpec(seed=seed, seats=["fake/llm", "cooperator", "conqueror", "turtle"]),
        [policy, build_policy("cooperator"), build_policy("conqueror"), build_policy("turtle")],
        config=cfg,
    )
    return backend, policy, result


def test_without_memory_no_history_is_ever_sent():
    backend, policy, _result = _play(memory=False)
    assert backend.calls, "aucun appel"
    assert all(call["history"] == [] for call in backend.calls)
    assert MEMORY_NOTE not in backend.calls[0]["system"]
    assert policy.report()["memory"] is False


def test_with_memory_each_history_is_a_prefix_of_the_next():
    """La propriété dont dépend le cache : on n'ajoute qu'à la fin."""
    backend, _policy, _result = _play(memory=True)
    histories = [call["history"] for call in backend.calls]
    assert histories[0] == []
    for before, after in zip(histories, histories[1:]):
        assert after[: len(before)] == before, "l'historique a été réécrit"
        assert len(after) == len(before) + 2, "un tour = un échange user/assistant"


def test_the_replayed_user_message_is_the_one_that_was_sent():
    """Rejouer un texte différent casserait le cache et mentirait au modèle."""
    backend, _policy, _result = _play(memory=True)
    for previous, current in zip(backend.calls, backend.calls[1:]):
        replayed = current["history"][-2]
        assert replayed == {"role": "user", "content": previous["user"]}


def test_the_replayed_answer_is_the_models_own_plan_in_stable_json():
    backend, _policy, _result = _play(memory=True)
    answer = backend.calls[1]["history"][-1]
    assert answer["role"] == "assistant"
    parsed = json.loads(answer["content"])
    assert parsed["reasoning"] == "tour 0"
    assert [move["index"] for move in parsed["plan"]] == [
        int(re.findall(r"\[(\d+)\]", backend.calls[0]["user"])[0])
    ]
    # Sérialisation déterministe : mêmes octets pour le même contenu.
    assert answer["content"] == json.dumps(parsed, ensure_ascii=False, sort_keys=True)


def test_a_mute_turn_is_remembered_as_lost_not_skipped():
    backend, policy, _result = _play(memory=True, mute_turns=(2,))
    # Un tour muet fait plusieurs appels (relances) : on lit le premier du tour 3.
    next_turn = next(call for call in backend.calls if re.search(r"TOUR 3/", call["user"]))
    remembered = [m["content"] for m in next_turn["history"] if m["role"] == "assistant"]
    assert remembered[2] == MUTE_MEMORY, "le tour 2 doit figurer, comme perdu"
    assert len(remembered) == 3
    assert policy.stats.mute_turns == 1


def test_memory_is_announced_in_the_system_prompt_only_when_on():
    backend, _policy, _result = _play(memory=True)
    assert MEMORY_NOTE in backend.calls[0]["system"]
    # Le système reste identique d'un appel à l'autre — sinon rien ne se cache.
    assert len({call["system"] for call in backend.calls}) == 1


def test_memory_never_crosses_two_games():
    cfg = base_config()
    catalog = build_catalog(cfg)
    backend = ScriptedBackend()
    policy = LLMPolicy(backend, cfg, catalog, memory=True)
    for seed in (1, 2):
        play_match(
            MatchSpec(seed=seed, seats=["fake/llm", "cooperator", "conqueror", "turtle"]),
            [policy, build_policy("cooperator"), build_policy("conqueror"),
             build_policy("turtle")],
            config=cfg,
        )
    first_of_second_game = next(
        call for call in backend.calls[1:] if call["history"] == []
    )
    assert first_of_second_game is not None


# ── registre : variantes et validation ──────────────────────────────────────

def test_every_declared_entry_is_valid():
    for seat, entry in load_models().items():
        validate_entry(seat, entry)


def test_memory_study_variants_point_to_a_declared_model():
    models = load_models()
    variants = {seat: entry for seat, entry in models.items() if entry.get("model")}
    assert variants, "aucune variante d'étude déclarée"
    for seat, entry in variants.items():
        assert api_model(seat, entry) != seat
        assert entry["model"] in models, (
            f"'{seat}' vise {entry['model']!r}, absent du registre : la variante "
            "n'aurait pas de témoin sans mémoire"
        )


def test_a_typo_in_an_entry_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError, match="memroy"):
        validate_entry("x", {"provider": "openrouter", "memroy": True})
    with pytest.raises(ValueError, match="true ou false"):
        validate_entry("x", {"provider": "openrouter", "memory": "oui"})
    assert "memory" in ENTRY_KEYS


def test_memory_defaults_to_the_global_setting():
    cfg = base_config()
    assert memory_enabled({}, cfg) is bool(cfg.llm.memory)
    assert memory_enabled({"memory": True}, cfg) is True


def test_factory_builds_a_variant_with_its_own_seat_and_memory(monkeypatch):
    from controla.policies.factory import build_policies

    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    cfg = base_config()
    plain, remembering = build_policies(
        ["openai/gpt-5-nano", "openai/gpt-5-nano@memoire"], cfg, build_catalog(cfg),
    )
    assert plain.memory is False and remembering.memory is True
    assert plain.backend.model == remembering.backend.model == "openai/gpt-5-nano"
    assert remembering.name == "openai/gpt-5-nano@memoire"
    # Deux backends : chaque variante garde ses propres compteurs de tokens.
    assert plain.backend is not remembering.backend
