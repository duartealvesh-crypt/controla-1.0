"""`controla.api` : le seul point d'entrée du moteur pour l'outillage externe."""

from __future__ import annotations

import json

import pytest

from controla import api

PANEL = ["cooperator", "conqueror", "expansionist", "turtle"]


def test_result_is_plain_json_data():
    result = api.play_game(PANEL, seed=3)
    assert json.loads(json.dumps(result)) == result
    assert len(result["history"]) == 15
    assert "board" in result["history"][0], "les instantanés de plateau sont toujours là"


def test_an_archetype_given_as_params_plays_exactly_like_its_name():
    """L'optimiseur passera des dicts : ils doivent valoir le nom, coup pour coup."""
    by_name = api.play_game(PANEL, seed=5)
    by_params = api.play_game([api.archetype_params(n) for n in PANEL], seed=5)
    by_name.pop("wall_seconds")
    by_params.pop("wall_seconds")
    assert by_name == by_params


def test_overrides_reach_the_engine():
    small = api.play_game(PANEL, seed=1, overrides={"world": {"grid_size": 9}})
    assert small["world"]["size"] == 9


def test_four_seats_are_required():
    with pytest.raises(ValueError, match="4 sièges"):
        api.play_game(PANEL[:3], seed=1)


def test_unknown_archetype_is_an_error():
    with pytest.raises(KeyError):
        api.archetype_params("pacifist")
