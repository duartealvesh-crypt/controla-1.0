from __future__ import annotations

import pytest

from controla import api
from lab.strategy import CLIP, Space


@pytest.mark.parametrize("name", ["cooperator", "conqueror", "turtle", "coalition_builder"])
def test_an_archetype_survives_encode_decode(name):
    space = Space()
    params = api.archetype_params(name)
    decoded = space.decode(space.encode(params), name)
    for key, weight in params["effect_weights"].items():
        assert decoded["effect_weights"][key] == pytest.approx(weight)
    for key, weight in params["kind_weights"].items():
        assert decoded["kind_weights"][key] == pytest.approx(weight)
    rate = params["commitment_rate"]
    assert decoded["propose_rate"] == pytest.approx(rate, abs=2e-3)
    assert decoded["accept_rate"] == pytest.approx(rate, abs=2e-3)


def test_any_vector_decodes_to_a_playable_archetype():
    space = Space()
    params = space.decode([CLIP * 3] * space.dim, "extrême")
    result = api.play_game([params, "cooperator", "conqueror", "turtle"], seed=2)
    assert result["seats"][0] == "extrême"


def test_frozen_diplomacy_removes_diplomatic_coordinates():
    free, frozen = Space(), Space(frozen_diplomacy=True)
    assert frozen.dim == free.dim - len(free.commitment_kinds) - 3
    params = frozen.decode([0.0] * frozen.dim, "x")
    assert params["propose_rate"] == 0.0 and params["accept_rate"] == 0.0
    adapted = frozen.adapt(api.archetype_params("cooperator"))
    assert adapted["propose_rate"] == 0.0 and adapted["accept_rate"] == 0.0


def test_unknown_target_rule_is_refused():
    with pytest.raises(ValueError):
        Space(target_rule="vengeance")
