"""Pacte convenu d'avance entre deux sièges (tests de déviation à deux du labo)."""

from __future__ import annotations

import pytest

from controla import api

SEEDS = range(1, 7)


def _member(partner: int, pact: str = "econ_alliance") -> dict:
    # Une tortue qui ne propose ni n'accepte rien d'elle-même : tout engagement
    # formé avec son partenaire vient donc du pacte.
    params = api.archetype_params("turtle")
    params.update(name="membre", propose_rate=0.0, accept_rate=0.0,
                  partners=[partner], pact=pact)
    return params


def _pair_commitments(seats, kind: str) -> int:
    return sum(
        1
        for seed in SEEDS
        for turn in api.play_game(seats, seed)["history"]
        for event in turn["events"]
        if event["kind"] == "commitment_formed" and event["commitment"] == kind
        and sorted(event["parties"]) == [0, 1]
    )


@pytest.mark.parametrize("pact", ["econ_alliance", "war_alliance"])
def test_a_pact_pair_seals_its_commitment(pact):
    without = [dict(_member(1, pact), partners=[]), dict(_member(0, pact), partners=[]),
               "conqueror", "expansionist"]
    with_pact = [_member(1, pact), _member(0, pact), "conqueror", "expansionist"]
    assert _pair_commitments(without, pact) == 0
    assert _pair_commitments(with_pact, pact) > 0


def test_no_pact_keeps_the_archetype_unchanged():
    params = api.archetype_params("cooperator")
    assert params["partners"] == [] or params["partners"] == ()
    first = api.play_game([params, "conqueror", "turtle", "expansionist"], seed=3)
    second = api.play_game(["cooperator", "conqueror", "turtle", "expansionist"], seed=3)
    first.pop("wall_seconds")
    second.pop("wall_seconds")
    first["seats"] = second["seats"]
    assert first == second


def _cooperator_pair() -> list:
    coop = api.archetype_params("cooperator")
    return [dict(coop, name="a", partners=[2]), "turtle",
            dict(coop, name="b", partners=[0]), "turtle"]


def test_an_active_commitment_is_not_proposed_again():
    """Plus d'écho : chaque promesse d'un allié n'appelle plus une réponse."""
    for seed in (101, 102):
        history = api.play_game(_cooperator_pair(), seed)["history"]
        pledges = sum(
            1 for turn in history for event in turn["events"]
            if event["kind"] == "negotiation_delivered" and event["faction"] in (0, 2)
            and event["target"] in (0, 2) and event["pledge"] == "econ_alliance"
        )
        assert pledges <= 15, pledges          # 6 à 7 par tour et par dieu avant


def test_an_economic_alliance_trades_the_turn_after_it_is_signed():
    """Le plancher de l'alliance tient après le déclin de la relation."""
    checked = 0
    for seed in (101, 102, 103):
        history = api.play_game(_cooperator_pair(), seed)["history"]
        for now, after in zip(history, history[1:]):
            if any(e["kind"] == "commitment_formed" and sorted(e["parties"]) == [0, 2]
                   for e in now["events"]):
                checked += 1
                assert any(e["kind"] == "trade" and sorted(e["factions"]) == [0, 2]
                           for e in after["events"]), f"seed {seed}, tour {after['turn']}"
    assert checked > 0
