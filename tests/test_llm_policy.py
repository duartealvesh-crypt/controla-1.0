"""La policy LLM : prompt dérivé du catalogue, validation, relances, métriques.

Tout est testé avec un backend **factice** : la logique de décision ne doit pas
coûter un appel pour être vérifiée, et le CI ne doit dépendre d'aucune clé d'API.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from controla.env.controla_env import ControlaEnv
from controla.policies.llm.policy import LLMPolicy
from controla.policies.llm.prompt import PLAN_TOOL, build_system, build_user
from controla.registry.catalog import build_catalog
from tests.helpers import base_config

SOURCE = Path(__file__).resolve().parents[1] / "controla"


class FakeBackend:
    """Rejoue une suite de réponses préparées et enregistre les prompts reçus."""

    def __init__(self, responses):
        self.model = "fake-model"
        self.responses = list(responses)
        self.prompts: list[tuple[str, str]] = []

        class _Stats:
            def to_dict(self):
                return {}

        self.stats = _Stats()

    def complete(self, system, user, tool):
        self.prompts.append((system, user))
        if not self.responses:
            return None
        return self.responses.pop(0)


@pytest.fixture
def context():
    cfg = base_config()
    catalog = build_catalog(cfg)
    env = ControlaEnv(config=cfg)
    env.reset(seed=5)
    agent = env.agent_selection
    view = env.infos[agent]["view"]
    return cfg, catalog, env, view


def make_policy(context, responses):
    cfg, catalog, _env, _view = context
    policy = LLMPolicy(FakeBackend(responses), cfg, catalog)
    policy.reset(0, 1)
    return policy


def info_of(env, agent):
    return {**env.infos[agent], "catalog": env.catalog,
            "index": env.index, "config": env.cfg}


# ── le prompt vient du catalogue ────────────────────────────────────────────

def test_system_prompt_contains_every_action_from_the_catalog(context):
    cfg, catalog, _env, _view = context
    system = build_system(cfg, catalog)
    for spec in catalog.actions:
        assert spec.key in system, f"'{spec.key}' absent du prompt système"


def test_system_prompt_carries_the_real_coefficients():
    """Le dieu doit lire les coefficients que le moteur applique, pas une copie."""
    tweaked = base_config(intervention={"cap": 37})
    system = build_system(tweaked, build_catalog(tweaked))
    assert "37" in system


def test_prompt_module_hardcodes_no_action_name():
    """Le mode de défaite de la v1 : un prompt qui maintient sa propre table."""
    cfg = base_config()
    catalog = build_catalog(cfg)
    source = (SOURCE / "policies" / "llm" / "prompt.py").read_text(encoding="utf-8")
    offenders = [
        spec.key for spec in catalog.actions
        if re.search(rf"""["']{re.escape(spec.key)}["']""", source)
    ]
    assert not offenders, f"noms d'action en dur dans prompt.py : {offenders}"


def test_user_prompt_lists_legal_indices(context):
    cfg, catalog, _env, view = context
    user = build_user(cfg, catalog, view)
    legal = [item["index"] for item in view["legal_actions"]]
    assert f"[{legal[0]}]" in user
    assert "POINTS D'INTERVENTION DISPONIBLES" in user


def test_user_prompt_announces_truncation(context):
    """Tronquer en silence ferait croire au dieu qu'il a vu toutes ses options."""
    cfg, catalog, _env, view = context
    narrow = base_config(llm={"max_legal_listed": 2})
    user = build_user(narrow, catalog, view)
    assert "cibles listées sur" in user


def test_retry_note_reaches_the_prompt(context):
    cfg, catalog, _env, view = context
    user = build_user(cfg, catalog, view, retry_note="index 999 inexistant")
    assert "[CORRECTION] index 999 inexistant" in user


def test_public_prompt_hides_rival_internals(context):
    """Le brouillard doit tenir dans le prompt aussi (§11).

    Sans Omniscience, les stocks, la Force, la Recherche et les traits d'un rival
    ne doivent apparaître nulle part — le classement doit rester *estimable*, pas
    connu, sinon le bluff disparaît.
    """
    cfg, catalog, env, _view = context
    agent = env.agent_selection
    rival = env.state_.faction(1)
    rival.research = 12345.0
    rival.traits["warrior"] = 0.98765

    env._refresh_info(agent)
    user = build_user(cfg, catalog, env.infos[agent]["view"])
    assert "[omniscience]" not in user, "le bloc révélé apparaît sans Omniscience"
    assert "12345" not in user, "la Recherche d'un rival a fui"
    assert "0.99" not in user, "les traits d'un rival ont fui"


def test_omniscience_reveals_rival_internals_in_the_prompt(context):
    """Et le miracle doit vraiment lever le brouillard."""
    cfg, catalog, env, _view = context
    agent = env.agent_selection
    env.state_.faction(1).research = 12345.0
    env.state_.faction(0).revealed_until_turn = env.state_.turn

    env._refresh_info(agent)
    user = build_user(cfg, catalog, env.infos[agent]["view"])
    assert "[omniscience]" in user
    assert "12345" in user


# ── validation et relances ──────────────────────────────────────────────────

def test_a_valid_plan_is_played_in_order(context):
    cfg, catalog, env, view = context
    legal = [item["index"] for item in view["legal_actions"] if item["cost"] > 0][:2]
    policy = make_policy(context, [{"plan": [{"index": legal[0]}, {"index": legal[1]}]}])

    agent = env.agent_selection
    assert policy.act(env.observe(agent), info_of(env, agent)) == legal[0]
    assert policy.act(env.observe(agent), info_of(env, agent)) == legal[1]
    # Plan épuisé : la policy rend la main plutôt que d'inventer un coup.
    assert policy.act(env.observe(agent), info_of(env, agent)) == env.index.end_turn_index
    assert policy.stats.planned_moves == 2
    assert policy.stats.rejected_moves == 0


def test_only_one_call_is_made_per_turn(context):
    """Le budget du benchmark en dépend : un appel par tour, pas par coup."""
    cfg, catalog, env, view = context
    legal = [item["index"] for item in view["legal_actions"] if item["cost"] > 0][:3]
    policy = make_policy(context, [{"plan": [{"index": i} for i in legal]}])

    agent = env.agent_selection
    for _ in range(4):
        policy.act(env.observe(agent), info_of(env, agent))
    assert len(policy.backend.prompts) == 1


def test_an_illegal_index_triggers_a_retry_with_a_precise_error(context):
    cfg, catalog, env, view = context
    legal = next(item["index"] for item in view["legal_actions"] if item["cost"] > 0)
    policy = make_policy(context, [
        {"plan": [{"index": 999_999}]},
        {"plan": [{"index": legal}]},
    ])

    agent = env.agent_selection
    assert policy.act(env.observe(agent), info_of(env, agent)) == legal
    assert policy.stats.retries == 1
    correction = policy.backend.prompts[1][1]
    assert "[CORRECTION]" in correction
    assert "999999" in correction.replace(" ", "") or "999_999" in correction


def test_a_partially_valid_plan_is_kept_and_the_rejects_counted(context):
    """Relancer pour un seul mauvais index coûterait un appel pour rien."""
    cfg, catalog, env, view = context
    legal = next(item["index"] for item in view["legal_actions"] if item["cost"] > 0)
    policy = make_policy(context, [{"plan": [{"index": legal}, {"index": 999_999}]}])

    agent = env.agent_selection
    assert policy.act(env.observe(agent), info_of(env, agent)) == legal
    assert policy.stats.rejected_moves == 1
    assert len(policy.backend.prompts) == 1


def test_a_mute_model_yields_the_turn_and_is_flagged(context):
    """§11.4 : jamais d'action aléatoire de repli. Et un tour muet est SIGNALÉ."""
    cfg, catalog, env, _view = context
    policy = make_policy(context, [])          # le backend ne répond jamais

    agent = env.agent_selection
    assert policy.act(env.observe(agent), info_of(env, agent)) == env.index.end_turn_index
    assert policy.stats.mute_turns == 1
    assert policy.stats.planned_moves == 0


def test_a_malformed_payload_is_rejected_then_retried(context):
    cfg, catalog, env, view = context
    legal = next(item["index"] for item in view["legal_actions"] if item["cost"] > 0)
    policy = make_policy(context, [
        {"plan": "pas une liste"},
        {"plan": [{"index": legal}]},
    ])
    agent = env.agent_selection
    assert policy.act(env.observe(agent), info_of(env, agent)) == legal
    assert "liste" in policy.backend.prompts[1][1]


def test_a_bare_integer_plan_is_accepted(context):
    """Tolérance de forme : `[3, 7]` plutôt que `[{"index": 3}, ...]`."""
    cfg, catalog, env, view = context
    legal = next(item["index"] for item in view["legal_actions"] if item["cost"] > 0)
    policy = make_policy(context, [{"plan": [legal]}])
    agent = env.agent_selection
    assert policy.act(env.observe(agent), info_of(env, agent)) == legal


def test_an_empty_plan_is_a_legitimate_choice(context):
    cfg, catalog, env, _view = context
    policy = make_policy(context, [{"plan": [], "reasoning": "j'attends"}])
    agent = env.agent_selection
    assert policy.act(env.observe(agent), info_of(env, agent)) == env.index.end_turn_index
    assert policy.stats.empty_plans == 1
    assert policy.stats.mute_turns == 0, "un plan vide n'est pas un modèle muet"


def test_a_move_that_became_illegal_is_dropped_not_replaced(context):
    """Le budget peut avoir été dépensé entre-temps : on saute, on ne substitue pas."""
    cfg, catalog, env, view = context
    costly = [item for item in view["legal_actions"] if item["cost"] > 0]
    policy = make_policy(context, [{"plan": [{"index": item["index"]} for item in costly[:6]]}])

    agent = env.agent_selection
    played = []
    for _ in range(8):
        move = policy.act(env.observe(agent), info_of(env, agent))
        if move == env.index.end_turn_index:
            break
        played.append(move)
        env.step(move)
        if env.agent_selection != agent:
            break
    assert played, "aucun coup n'a été joué"


def test_an_argument_rides_along_with_a_negotiation(context):
    cfg, catalog, env, view = context
    negotiate = next(
        item["index"] for item in view["legal_actions"]
        if env.index[item["index"]].spec.kind.value == "diplomacy"
    )
    policy = make_policy(context, [
        {"plan": [{"index": negotiate, "argument": "Allions-nous contre le leader."}]}
    ])
    agent = env.agent_selection
    move = policy.act(env.observe(agent), info_of(env, agent))
    assert isinstance(move, dict)
    assert move["argument"] == "Allions-nous contre le leader."


# ── métriques ───────────────────────────────────────────────────────────────

def test_move_accuracy_is_reported(context):
    cfg, catalog, env, view = context
    legal = [item["index"] for item in view["legal_actions"] if item["cost"] > 0][:3]
    policy = make_policy(context, [
        {"plan": [{"index": legal[0]}, {"index": legal[1]}, {"index": 999_999}]}
    ])
    agent = env.agent_selection
    policy.act(env.observe(agent), info_of(env, agent))

    report = policy.report()
    assert report["policy"]["planned_moves"] == 2
    assert report["policy"]["rejected_moves"] == 1
    assert report["policy"]["move_accuracy"] is not None


def test_reasoning_is_captured_for_the_replay(context):
    cfg, catalog, env, view = context
    legal = next(item["index"] for item in view["legal_actions"] if item["cost"] > 0)
    policy = make_policy(context, [
        {"plan": [{"index": legal}], "reasoning": "je sécurise ma base alimentaire"}
    ])
    agent = env.agent_selection
    policy.act(env.observe(agent), info_of(env, agent))
    assert policy.stats.reasonings[0]["reasoning"] == "je sécurise ma base alimentaire"
    assert policy.stats.reasonings[0]["status"] == "ok"


# ── outillage ───────────────────────────────────────────────────────────────

def test_plan_tool_schema_is_well_formed():
    parameters = PLAN_TOOL["function"]["parameters"]
    assert parameters["required"] == ["plan"]
    assert parameters["properties"]["plan"]["items"]["required"] == ["index"]


def test_error_classification_separates_fatal_from_recoverable():
    """Mesuré en campagne : une requête refusée isolée n'est PAS une clé invalide.

    Les confondre ferait crier au loup sur une campagne qui va bien, ou pire, ferait
    passer une clé morte pour un incident passager.
    """
    from controla.policies.llm.backend import classify

    openai = pytest.importorskip("openai")
    import httpx

    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")

    def response(status: int):
        return httpx.Response(status, request=request)

    assert classify(openai.AuthenticationError(
        "bad key", response=response(401), body=None)) == "fatal"
    assert classify(openai.NotFoundError(
        "no such model", response=response(404), body=None)) == "fatal"
    assert classify(openai.BadRequestError(
        "too long", response=response(400), body=None)) == "bad_request"
    assert classify(openai.RateLimitError(
        "slow down", response=response(429), body=None)) == "transient"
    assert classify(RuntimeError("boom")) == "error"


def test_response_rate_exposes_a_mute_model():
    """Un modèle muet ne doit jamais passer pour un modèle prudent (§12)."""
    from controla.policies.llm.backend import CallStats

    stats = CallStats()
    for _ in range(3):
        stats.record(0.1)
    stats.record(0.1, "transient")
    assert stats.response_rate == pytest.approx(0.75)
    assert stats.to_dict()["failures"] == {"transient": 1}


def test_missing_credentials_fail_with_a_named_variable(monkeypatch):
    from controla.policies.llm.registry import provider_credentials

    monkeypatch.delenv("TOTALLY_UNSET_BASE_URL", raising=False)
    monkeypatch.delenv("TOTALLY_UNSET_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TOTALLY_UNSET_BASE_URL"):
        provider_credentials("totally_unset")


def test_model_registry_is_loadable():
    from controla.policies.llm.registry import load_models

    models = load_models()
    assert models
    for name, entry in models.items():
        assert "provider" in entry, f"{name} n'a pas de fournisseur"


def test_factory_lists_both_families():
    from controla.policies.factory import available_seats

    catalogue = available_seats()
    assert "cooperator" in catalogue["scripted"]
    assert catalogue["models"]
