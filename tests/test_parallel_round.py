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
