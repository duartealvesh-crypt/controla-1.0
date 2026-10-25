"""Le banc de sélection : des sondes identiques pour tous, un jugement juste.

Un banc qui juge mal choisit mal, et un mauvais choix se paie ensuite en
campagnes entières. On vérifie donc qu'il distingue bien « ne répond pas »,
« répond de travers », « triche sur les règles » et « joue proprement ».
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

from controla.policies.llm import bench
from controla.policies.llm.backend import LLMBackend
from controla.registry.catalog import build_catalog
from tests.helpers import base_config


def _probe(costs=None, points=6):
    return bench.Probe(seed=1, turn=0, system="s", user="u",
                       costs=costs or {3: 2, 7: 1, 9: 3}, points=points)


# ── jugement ────────────────────────────────────────────────────────────────

def test_judge_separates_each_failure_mode():
    probe = _probe()
    assert bench.judge(None, probe)["answered"] is False
    assert bench.judge({"plan": "pas une liste"}, probe)["well_formed"] is False
    assert bench.judge({"plan": [{"index": "3"}]}, probe)["well_formed"] is False

    illegal = bench.judge({"plan": [{"index": 3}, {"index": 99}]}, probe)
    assert illegal["well_formed"] and illegal["legal_rate"] == 0.5

    over = bench.judge({"plan": [3, 9, 7]}, probe)                # 2 + 3 + 1 = 6 ≤ 6
    assert over["within_budget"] is True
    broke = bench.judge({"plan": [3, 9, 9]}, _probe(points=6))   # 2 + 3 + 3 = 8 > 6
    assert broke["within_budget"] is False


def test_an_empty_plan_is_well_formed_but_flagged():
    verdict = bench.judge({"plan": [], "reasoning": "j'attends"}, _probe())
    assert verdict["well_formed"] and verdict["empty_plan"]
    assert verdict["legal_rate"] is None                         # rien à juger


# ── sondes ──────────────────────────────────────────────────────────────────

def test_probes_are_real_turns_and_reproducible():
    cfg = base_config()
    catalog = build_catalog(cfg)
    first = bench.build_probes(cfg, catalog, seeds=(1,), turns=(0, 6))
    again = bench.build_probes(cfg, catalog, seeds=(1,), turns=(0, 6))
    assert [p.turn for p in first] == [0, 6]
    assert first == again, "deux bancs à un mois d'écart doivent lire les mêmes prompts"
    for probe in first:
        assert probe.costs and probe.points > 0
        listed = {int(i) for i in re.findall(r"\[(\d+)\]", probe.user)}
        assert listed <= set(probe.costs), "le prompt propose un index que le juge refuserait"


# ── découverte ──────────────────────────────────────────────────────────────

def test_discovery_keeps_only_free_reliable_tool_models():
    catalogue = {"data": [
        {"id": "a/bon:free", "supported_parameters": ["tools", "tool_choice"]},
        {"id": "a/instable:free", "supported_parameters": ["tools", "tool_choice"]},
        {"id": "a/sans-choix:free", "supported_parameters": ["tools"]},
        {"id": "a/payant", "supported_parameters": ["tools", "tool_choice"],
         "pricing": {"prompt": "0.000001", "completion": "0.000001"}},
        {"id": "openrouter/free", "supported_parameters": ["tools", "tool_choice"]},
    ]}
    uptime = {"a/bon:free": 99.0, "a/instable:free": 23.0}

    def fetch(url):
        if url == bench.CATALOGUE_URL:
            return catalogue
        model = url.split("/models/")[1].rsplit("/endpoints", 1)[0]
        return {"data": {"endpoints": [{"uptime_last_1d": uptime.get(model),
                                        "max_completion_tokens": 8192}]}}

    found = [entry["id"] for entry in bench.discover_free_candidates(90.0, fetch=fetch)]
    assert found == ["a/bon:free"]


# ── banc complet, client simulé ─────────────────────────────────────────────

class _Completions:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def create(self, **kwargs):
        user = kwargs["messages"][-1]["content"]
        legal = [int(i) for i in re.findall(r"\[(\d+)\]", user)]
        arguments = self.behaviour(legal)
        message = SimpleNamespace(
            content=None,
            tool_calls=None if arguments is None else
            [SimpleNamespace(function=SimpleNamespace(arguments=arguments))],
        )
        usage = SimpleNamespace(prompt_tokens=6000, completion_tokens=200,
                                prompt_tokens_details=None, completion_tokens_details=None)
        finish = "length" if arguments is None else "tool_calls"
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                               usage=usage)


def _fake_backend(name, behaviour):
    backend = LLMBackend(base_url="http://local", api_key="x", model=name, max_retries=0)
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(behaviour)))
    return backend


def test_bench_ranks_a_compliant_model_above_a_broken_one():
    cfg = base_config()
    probes = bench.build_probes(cfg, build_catalog(cfg), seeds=(1,), turns=(0, 6))
    backends = {
        "propre": _fake_backend("propre", lambda legal: json.dumps(
            {"reasoning": "ok", "plan": [{"index": legal[0]}]})),
        "tricheur": _fake_backend("tricheur", lambda legal: json.dumps(
            {"reasoning": "ok", "plan": [{"index": 999_999}]})),
        "muet": _fake_backend("muet", lambda legal: None),
    }
    summaries = bench.rank(bench.run_bench(backends, probes, requests_per_minute=0))
    assert [s["model"] for s in summaries] == ["propre", "tricheur", "muet"]

    propre, tricheur, muet = summaries
    assert propre["well_formed"] == 1.0 and propre["legal_rate"] == 1.0
    assert tricheur["well_formed"] == 1.0 and tricheur["legal_rate"] == 0.0
    assert muet["answered"] == 0.0 and muet["errors"] == {"truncated": len(probes)}
    assert propre["prompt_tokens"] == 6000 * len(probes)


def test_bench_paces_itself_on_the_rate_limit():
    """20 requêtes/minute : au moins 3 s entre deux appels, quel que soit le modèle."""
    cfg = base_config()
    probes = bench.build_probes(cfg, build_catalog(cfg), seeds=(1,), turns=(0,))
    backends = {name: _fake_backend(name, lambda legal: json.dumps({"plan": []}))
                for name in ("a", "b", "c")}
    now = [0.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    bench.run_bench(backends, probes, requests_per_minute=20,
                    sleep=sleep, clock=lambda: now[0])
    assert waits == [3.0, 3.0]                         # 3 appels, 2 attentes
