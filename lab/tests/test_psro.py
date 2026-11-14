"""La gestion de la table par PSRO, avec un faux runner (aucune partie jouée)."""

from __future__ import annotations

import numpy as np

from lab.metagame import PayoffTable
from lab.psro import (Settings, _layout, fill_table, record_cases, refine,
                       scenarios, score_against)


class FakeRunner:
    def __init__(self):
        self.games = 0

    def play(self, games):
        self.games += len(games)
        return [[3, 2, 1, 0] for _ in games]


POPULATION = [{"params": {"name": str(i)}} for i in range(3)]


def test_base_fill_covers_every_profile_with_rotations():
    table, runner = PayoffTable(), FakeRunner()
    played = fill_table(runner, table, POPULATION, Settings(base_games=4), np.random.default_rng(0))
    assert table.missing(3) == []
    assert played == 15 * 4
    assert table.count((0, 0, 0, 0), 0) == 16          # 4 parties × 4 sièges


def test_refine_tops_up_only_the_cells_that_weigh_under_sigma():
    table, runner = PayoffTable(), FakeRunner()
    s = Settings(base_games=4, refine_target=20, refine_pmin=0.5)
    rng = np.random.default_rng(0)
    fill_table(runner, table, POPULATION, s, rng)
    refine(runner, table, POPULATION, np.array([1.0, 0.0, 0.0]), s, rng)
    # Sous σ = 100 % stratégie 0, seul le trio (0, 0, 0) pèse.
    assert table.count((0, 0, 0, 1), 1) >= 20
    assert table.count((0, 0, 0, 2), 2) >= 20
    assert table.count((0, 1, 1, 2), 2) == 4           # case improbable : intacte
    assert refine(runner, table, POPULATION, np.array([1.0, 0.0, 0.0]), s, rng) == 0


def test_evaluation_games_are_recorded_with_the_new_index():
    table = PayoffTable()
    games = [((7, [0, 1, 1], [2]), [0, 1, 3, 2])]     # le candidat était au siège 2
    record_cases(table, 2, games)
    assert table.payoff((0, 1, 1, 2), 2) == 3.0
    assert table.payoff((0, 1, 1, 2), 1) == 1.5        # (1 + 2) / 2


def test_team_games_put_the_candidate_on_both_seats():
    table = PayoffTable()
    games = [((7, [0, 1], [1, 3]), [0, 3, 1, 2])]     # équipe aux sièges 1 et 3
    record_cases(table, 2, games)
    assert table.payoff((0, 1, 2, 2), 2) == 2.5        # (3 + 2) / 2
    assert table.count((0, 1, 2, 2), 2) == 2


def test_team_scenarios_draw_two_opponents_and_two_seats():
    cases = scenarios(np.random.default_rng(0), np.array([0.5, 0.5]), 20, copies=2)
    assert all(len(others) == 2 and len(positions) == 2 and positions[0] < positions[1]
               for _, others, positions in cases)
    seats = _layout("moi", ["a", "b"], [0, 2])
    assert seats == ["moi", "a", "moi", "b"]


def test_score_against_averages_the_team_seats():
    class Runner:
        games = 0

        def play(self, games):
            return [[3, 2, 1, 0] for _ in games]

    cases = [(1, [0, 0], [0, 1]), (2, [0, 0], [2, 3])]
    points = score_against(Runner(), [{"name": "x"}], cases, POPULATION)
    assert points.tolist() == [[2.5, 0.5]]


def test_the_loop_adds_solo_and_team_responses(monkeypatch, tmp_path):
    import lab.psro as psro
    from lab.strategy import Space

    class Pool(FakeRunner):
        def __init__(self, workers=None):
            super().__init__()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    def fake_oracle(runner, space, population, sigma, reference, s, rng, label, log, copies):
        cases = psro.scenarios(rng, sigma, 4, copies)
        games = list(zip(cases, [[3, 2, 1, 0]] * len(cases)))
        return dict(population[0]["params"], name=label), 2.9, 0.01, games

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(psro, "Runner", Pool)
    monkeypatch.setattr(psro, "oracle", fake_oracle)
    settings = psro.Settings(iterations=2, panel=["turtle", "conqueror"], refine_target=4)
    state = psro.run("t", Space(), settings, log=lambda line: None)
    assert [p["label"] for p in state["population"]] == [
        "turtle", "conqueror", "br1", "eq1", "br2", "eq2"]
    assert state["status"] == "max_iterations"
    assert state["iterations"][0]["accepted"] is True
    assert state["iterations"][0]["team_accepted"] is True
