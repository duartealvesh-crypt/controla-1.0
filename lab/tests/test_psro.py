"""La gestion de la table par PSRO, avec un faux runner (aucune partie jouée)."""

from __future__ import annotations

import numpy as np

from lab.metagame import PayoffTable
from lab.psro import Settings, fill_table, record_cases, refine


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
    games = [((7, [0, 1, 1], 2), [0, 1, 3, 2])]       # le candidat était au siège 2
    record_cases(table, 2, games)
    assert table.payoff((0, 1, 1, 2), 2) == 3.0
    assert table.payoff((0, 1, 1, 2), 1) == 1.5        # (1 + 2) / 2
