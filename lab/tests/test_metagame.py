"""Le méta-solveur sur des jeux synthétiques dont on connaît la réponse."""

from __future__ import annotations

import numpy as np
import pytest

from lab.metagame import PayoffTable, profiles, seatings


def _table(k: int, payoff) -> PayoffTable:
    """Table remplie par `payoff(i, autres)` au lieu de parties jouées."""
    table = PayoffTable()
    for profile in profiles(k):
        seats = list(profile)
        points = [payoff(i, seats[:n] + seats[n + 1:]) for n, i in enumerate(seats)]
        table.record(seats, points)
    return table


def test_profile_count_is_multiset_count():
    assert len(profiles(3)) == 15
    assert len(profiles(8)) == 330


def test_seatings_rotate_every_strategy_through_every_corner():
    rotated = seatings((0, 0, 1, 2))
    assert len(rotated) == 4
    assert all(sorted(r) == [0, 0, 1, 2] for r in rotated)


def test_a_dominant_strategy_takes_the_whole_mixture():
    # La stratégie 2 gagne 1 point de plus que les autres, quoi qu'il arrive.
    table = _table(3, lambda i, others: 1.0 + (1.0 if i == 2 else 0.0))
    sigma = table.replicator(3)
    assert sigma[2] == pytest.approx(1.0, abs=1e-3)
    assert table.regret(sigma) == pytest.approx(0.0, abs=1e-3)


def test_rock_paper_scissors_gives_the_uniform_mixture():
    beats = {(0, 1), (1, 2), (2, 0)}

    def payoff(i, others):
        return 1.5 + sum(0.5 if (i, j) in beats else -0.5 if (j, i) in beats else 0.0
                         for j in others)

    table = _table(3, payoff)
    sigma = table.replicator(3)
    assert sigma == pytest.approx(np.full(3, 1 / 3), abs=0.05)
    assert table.regret(sigma) < 0.05


def test_fitness_against_a_pure_mixture_reads_the_table():
    table = _table(2, lambda i, others: float(i) + 0.1 * sum(others))
    # Contre 100 % stratégie 1 : profil {i, 1, 1, 1}.
    f = table.fitness(np.array([0.0, 1.0]))
    assert f[0] == pytest.approx(0.0 + 0.3)
    assert f[1] == pytest.approx(1.0 + 0.3)


def test_table_survives_a_json_round_trip():
    table = _table(2, lambda i, others: float(i))
    copy = PayoffTable.from_json(table.to_json())
    assert copy.payoff((0, 0, 1, 1), 1) == table.payoff((0, 0, 1, 1), 1)
    assert copy.missing(2) == []
