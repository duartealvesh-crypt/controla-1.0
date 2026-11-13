"""Proposer et accepter, réglables séparément.

Sans cette séparation, un dieu qui ne propose jamais n'accepte jamais non plus :
il prive les autres de partenaire, et la diplomatie paraît inutile par simple
absence de réponse.
"""

from __future__ import annotations

from controla import api


def _formed(seats, seeds=range(1, 7)) -> int:
    return sum(
        1
        for seed in seeds
        for turn in api.play_game(seats, seed)["history"]
        for event in turn["events"]
        if event["kind"] == "commitment_formed"
    )


def _variant(propose: float, accept: float) -> dict:
    params = api.archetype_params("turtle")
    params.update(name="variante", propose_rate=propose, accept_rate=accept)
    return params


def test_a_listener_forms_commitments_a_deaf_god_does_not():
    listener = _variant(propose=0.0, accept=1.0)
    deaf = _variant(propose=0.0, accept=0.0)
    with_listeners = _formed(["cooperator", listener, listener, listener])
    with_deaf = _formed(["cooperator", deaf, deaf, deaf])
    assert with_listeners > with_deaf


def test_defaults_keep_the_historical_panel_unchanged():
    """`None` = `commitment_rate` pour les deux : les archétypes jouent comme avant."""
    params = api.archetype_params("cooperator")
    assert params["propose_rate"] is None and params["accept_rate"] is None
    explicit = dict(params, propose_rate=params["commitment_rate"],
                    accept_rate=params["commitment_rate"])
    seats_default = [params, "conqueror", "turtle", "expansionist"]
    seats_explicit = [explicit, "conqueror", "turtle", "expansionist"]
    first = api.play_game(seats_default, seed=4)
    second = api.play_game(seats_explicit, seed=4)
    first.pop("wall_seconds")
    second.pop("wall_seconds")
    assert first == second
