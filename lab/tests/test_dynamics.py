from __future__ import annotations

import pytest

from controla import api
from lab import dynamics

PANEL = ["cooperator", "conqueror", "expansionist", "turtle"]


def test_rotations_put_every_strategy_in_every_corner():
    rotated = dynamics.rotations(PANEL)
    for position in range(4):
        assert sorted(seats[position] for seats in rotated) == sorted(PANEL)


def test_spearman_bounds():
    assert dynamics.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert dynamics.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_contact_detects_touching_territories():
    # 3×3 : faction 0 en haut à gauche, faction 1 juste à sa droite.
    touching = [0, 1, -1, -1, -1, -1, -1, -1, -1]
    apart = [0, -1, 1, -1, -1, -1, -1, -1, -1]
    assert dynamics._contact(touching, 3)
    assert not dynamics._contact(apart, 3)


def test_overrides_parse_dotted_keys():
    assert dynamics._overrides(["world.grid_size=10", "combat.attrition_loser=0.3"]) == {
        "world": {"grid_size": 10}, "combat": {"attrition_loser": 0.3},
    }
    assert dynamics._overrides([]) is None


def test_metrics_of_a_real_game_are_consistent():
    game = dynamics.game_metrics(api.play_game(PANEL, seed=1), mid_turn=7)
    assert game["first_contact"] is None or 0 <= game["first_contact"] <= 15
    assert 0 <= game["war_pairs"] <= 6
    assert -1.0 <= game["mid_final_spearman"] <= 1.0
    assert sorted(seat["points"] for seat in game["per_seat"]) == [0, 1, 2, 3]
    assert [seat["name"] for seat in game["per_seat"]] == PANEL


def test_summary_aggregates_by_strategy():
    games = dynamics.run(PANEL, seeds=[1], workers=1)
    summary = dynamics.summarize(games)
    assert summary["games"] == 4
    # Chaque stratégie a joué chaque coin une fois.
    assert all(s["seats"] == 4 for s in summary["strategies"].values())
    total_points = sum(s["mean_points"] * s["seats"] for s in summary["strategies"].values())
    assert total_points == 4 * sum(dynamics.RANK_POINTS)
