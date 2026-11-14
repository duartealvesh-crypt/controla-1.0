"""Départs multiples, α-Rank et coalitions, sur des jeux dont on connaît la réponse."""

from __future__ import annotations

import numpy as np
import pytest

from controla import api
from lab.coalitions import deviation_test
from lab.equilibria import alpha_rank, multistart
from lab.metagame import PayoffTable, profiles
from lab.runner import Runner


def _table(k: int, payoff) -> PayoffTable:
    table = PayoffTable()
    for profile in profiles(k):
        seats = list(profile)
        table.record(seats, [payoff(i, seats[:n] + seats[n + 1:]) for n, i in enumerate(seats)])
    return table


def test_a_coordination_game_has_two_basins():
    # Gain = nombre d'adversaires qui jouent comme moi : « tous 0 » et « tous 1 »
    # sont deux équilibres stables.
    table = _table(2, lambda i, others: float(sum(j == i for j in others)))
    found = multistart(table, 2, starts=40, rng=np.random.default_rng(1))
    pure = sorted(int(np.argmax(item["sigma"])) for item in found if item["sigma"].max() > 0.99)
    assert pure == [0, 1]
    assert sum(item["share"] for item in found) == pytest.approx(1.0)


def test_alpha_rank_puts_the_mass_on_a_dominant_strategy():
    table = _table(3, lambda i, others: 1.0 + (1.0 if i == 2 else 0.0))
    pi = alpha_rank(table, 3, alpha=20.0)
    assert pi[2] > 0.99


def test_alpha_rank_is_uniform_on_rock_paper_scissors():
    beats = {(0, 1), (1, 2), (2, 0)}
    table = _table(3, lambda i, others: 1.5 + sum(
        0.5 if (i, j) in beats else -0.5 if (j, i) in beats else 0.0 for j in others))
    assert alpha_rank(table, 3, alpha=10.0) == pytest.approx(np.full(3, 1 / 3), abs=1e-6)


def test_deviation_test_plays_paired_games():
    population = [{"label": name, "params": api.archetype_params(name)}
                  for name in ("turtle", "conqueror")]
    # σ = 100 % turtle : la table n'a besoin que de dire que turtle domine.
    table = _table(2, lambda i, others: 2.0 if i == 0 else 1.0)
    state = {"name": "test", "population": population, "payoffs": table, "complete": 2}
    with Runner(workers=1) as runner:
        result = deviation_test(runner, state, "econ_alliance", games=3,
                                rng=np.random.default_rng(0))
        assert runner.games == 6
    assert result["games"] == 3
    assert 0.0 <= result["reference"] <= 3.0 and 0.0 <= result["with_pact"] <= 3.0
    assert 0.0 <= result["sealed"] <= 1.0
