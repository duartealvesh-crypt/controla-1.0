"""Planification parallèle des rondes : plus rapide, et rien d'autre.

Le parallélisme n'est acceptable que s'il ne change RIEN au jeu : mêmes prompts
envoyés à chaque dieu, mêmes coups, même partie. Sinon le classement
dépendrait d'un réglage de performance. Ces tests le vérifient octet par octet,
puis vérifient que les appels se chevauchent bel et bien.
"""

from __future__ import annotations

import re
import threading
import time
from types import SimpleNamespace

from controla.arena.play import MatchSpec, play_match
from controla.config.loader import load_config
from controla.policies.llm.backend import CallOutcome, CallStats, LLMBackend
from controla.policies.llm.policy import LLMPolicy
from controla.registry.catalog import build_catalog

SEATS = ["llm-a", "llm-b", "llm-c", "llm-d"]


class RecordingBackend:
    """Joue un coup déterministe, enregistre ses prompts, mesure le chevauchement."""

    in_flight = 0
    peak = 0
    _lock = threading.Lock()

    def __init__(self, name: str, delay: float = 0.0):
        self.model = self.label = name
        self.delay = delay
        self.users: list[str] = []
        self.stats = SimpleNamespace(to_dict=lambda: {})
        self.last_call = CallOutcome()

    def complete(self, system, user, tool, history=(), sink=None):
        cls = RecordingBackend
        with cls._lock:
            cls.in_flight += 1
            cls.peak = max(cls.peak, cls.in_flight)
        try:
            time.sleep(self.delay)
            self.users.append(user)
            legal = [int(i) for i in re.findall(r"\[(\d+)\]", user)]
            # Choix dépendant du siège et du tour, pour que les dieux divergent.
            pick = legal[(len(self.users) * 7 + len(self.model)) % len(legal)] if legal else None
            return {"reasoning": "r", "plan": [] if pick is None else [{"index": pick}]}
        finally:
            with cls._lock:
                cls.in_flight -= 1


def _play(parallel: bool, delay: float = 0.0, max_turns: int = 15):
    RecordingBackend.peak = 0
    cfg = load_config(overrides={"llm": {"parallel_round": parallel},
                                 "world": {"max_turns": max_turns}})
    catalog = build_catalog(cfg)
    backends = [RecordingBackend(name, delay) for name in SEATS]
    policies = [LLMPolicy(backend, cfg, catalog) for backend in backends]
    result = play_match(MatchSpec(seed=7, seats=SEATS), policies, config=cfg)
    return result, backends


def test_parallel_and_sequential_play_the_same_game():
    """Mêmes prompts, mêmes coups, même partie — au caractère près."""
    sequential, seq_backends = _play(parallel=False)
    parallel, par_backends = _play(parallel=True)

    for before, after in zip(seq_backends, par_backends):
        assert before.users == after.users, f"{before.model} n'a pas lu les mêmes prompts"
    assert parallel["history"] == sequential["history"]
    assert parallel["ranking"] == sequential["ranking"]


def test_calls_of_a_round_overlap_when_parallel():
    _play(parallel=False, delay=0.02, max_turns=3)
    assert RecordingBackend.peak == 1, "le séquentiel ne doit jamais chevaucher"

    started = time.perf_counter()
    _play(parallel=True, delay=0.2, max_turns=3)
    elapsed = time.perf_counter() - started
    assert RecordingBackend.peak == len(SEATS), "les 4 dieux doivent planifier ensemble"
    # 3 rondes × 0.2 s si parallèle, contre 2.4 s en séquentiel.
    assert elapsed < 1.5, f"pas de gain de temps mesurable ({elapsed:.2f} s)"


def test_a_shared_backend_loses_no_count_under_concurrency():
    """Deux sièges d'un même modèle partagent un backend, donc ses compteurs."""
    call = SimpleNamespace(function=SimpleNamespace(arguments='{"plan": []}'))
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call], content=None),
                                 finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1,
                              prompt_tokens_details=None, completion_tokens_details=None),
    )
    backend = LLMBackend(base_url="http://local", api_key="x", model="m")
    backend._client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: response)))

    sinks = [CallStats() for _ in range(8)]
    outcomes = []

    def worker(sink):
        for _ in range(200):
            backend.complete("s", "u", {"type": "function", "function": {"name": "plan_turn"}},
                             sink=sink)
        outcomes.append(backend.last_call)

    threads = [threading.Thread(target=worker, args=(sink,)) for sink in sinks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert backend.stats.calls == 1600
    assert backend.stats.prompt_tokens == 16000
    assert backend.stats.finish_reasons == {"tool_calls": 1600}
    assert all(sink.calls == 200 for sink in sinks)
    # `last_call` est propre à chaque thread : huit objets distincts.
    assert len({id(outcome) for outcome in outcomes}) == 8


def _play_with_delays(parallel: bool, delays: list[float], max_turns: int = 3):
    cfg = load_config(overrides={"llm": {"parallel_round": parallel},
                                 "world": {"max_turns": max_turns}})
    catalog = build_catalog(cfg)
    policies = [LLMPolicy(RecordingBackend(name, delay), cfg, catalog)
                for name, delay in zip(SEATS, delays)]
    return play_match(MatchSpec(seed=7, seats=SEATS), policies, config=cfg)


def test_each_round_records_its_duration_and_slowest_seat():
    """La durée d'une ronde parallèle est celle de son siège le plus lent."""
    result = _play_with_delays(parallel=True, delays=[0.01, 0.01, 0.15, 0.01])
    timings = result["round_timings"]
    assert [t["turn"] for t in timings] == [0, 1, 2]
    for timing in timings:
        assert timing["slowest"] == "god_2"
        assert set(timing["per_seat"]) == {"god_0", "god_1", "god_2", "god_3"}
        assert timing["seconds"] < sum(timing["per_seat"].values())   # chevauchement
        assert timing["parallel"] is True
    # Chaque décision connaît aussi sa propre durée.
    reasoning = result["policy_reports"]["god_2"]["reasonings"][0]
    assert reasoning["seconds"] >= 0.15


def test_sequential_rounds_last_the_sum_of_their_seats():
    result = _play_with_delays(parallel=False, delays=[0.05] * 4, max_turns=2)
    for timing in result["round_timings"]:
        assert timing["parallel"] is False
        assert timing["seconds"] >= sum(timing["per_seat"].values()) * 0.95


def test_a_scripted_panel_records_no_round_timing():
    """Rien à attendre, rien à mesurer : le JSONL d'un sweep reste inchangé."""
    from tests.helpers import run_scripted_match

    assert "round_timings" not in run_scripted_match(seed=3)


def test_usage_summary_reports_tokens_losses_and_rounds():
    from controla.replay.render import render_llm_usage

    result = _play_with_delays(parallel=True, delays=[0.0] * 4, max_turns=2)
    summary = render_llm_usage(result)
    assert "god_0" in summary and "rondes : 2" in summary
