"""Les instantanés de restitution disent bien ce que le moteur a fait.

Le risque propre à cette couche est la **divergence silencieuse** : un replay
qui affiche une carte plausible mais fausse ne lève aucune erreur, et l'on tire
des conclusions d'une partie qu'on n'a pas jouée. Ces tests comparent donc
l'instantané à l'état réel, case par case, plutôt que de vérifier qu'il « a la
bonne forme ».
"""

from __future__ import annotations

import json

from controla.arena.storage import read_game, write_game
from controla.registry.catalog import build_catalog
from controla.replay.render import BUILDING_GLYPHS, building_glyph
from controla.replay.snapshot import NONE_INDEX, board_snapshot, world_snapshot
from tests.helpers import base_config, run_scripted_match, run_with_checks


def test_board_snapshot_matches_the_grid_cell_by_cell():
    """La carte affichée EST la carte jouée — pas une reconstruction plausible."""
    cfg = base_config()
    catalog = build_catalog(cfg)

    def check(env):
        state = env.state_
        board = board_snapshot(env.cfg, env.catalog, state)
        size = state.grid.size
        assert len(board["owner"]) == size * size
        assert len(board["building"]) == size * size

        for y in range(size):
            for x in range(size):
                tile = state.grid.get(x, y)
                index = y * size + x
                expected_owner = NONE_INDEX if tile.owner is None else tile.owner
                assert board["owner"][index] == expected_owner, (
                    f"propriétaire divergent en ({x},{y}) au tour {state.turn}"
                )
                if tile.building is None:
                    assert board["building"][index] == NONE_INDEX
                else:
                    key = catalog.building_keys[board["building"][index]]
                    assert key == tile.building, (
                        f"bâtiment divergent en ({x},{y}) au tour {state.turn}"
                    )

    run_with_checks(seed=11, per_turn=check, config=cfg)


def test_faction_snapshot_matches_the_scoring_module():
    """Les chiffres de l'instantané viennent du moteur, pas d'un calcul parallèle."""
    from controla.dynamics.scoring import score, score_components

    cfg = base_config()

    def check(env):
        board = board_snapshot(env.cfg, env.catalog, env.state_)
        for entry in board["factions"]:
            faction_id = entry["id"]
            assert entry["score"] == round(score(cfg, env.state_, faction_id), 2)
            assert entry["tiles"] == env.state_.tile_count(faction_id)
            expected = score_components(env.state_, faction_id)
            for axis, value in entry["components"].items():
                assert value == round(expected[axis], 2)

    run_with_checks(seed=3, per_turn=check, config=cfg)


def test_snapshots_are_off_by_default():
    """Un sweep de milliers de parties ne doit pas payer la restitution."""
    result = run_scripted_match(seed=5)
    assert "world" not in result
    assert all("board" not in report for report in result["history"])


def test_snapshots_cover_every_turn_when_asked():
    """Chaque tour a son image, plus l'état initial — sinon le replay démarre
    après que le premier tour a déjà tout changé."""
    result = run_scripted_match(seed=5, snapshots=True)
    assert "initial_board" in result["world"]
    assert result["history"], "aucun tour joué"
    for report in result["history"]:
        assert "board" in report, f"tour {report['turn']} sans instantané"

    size = result["world"]["size"]
    assert len(result["world"]["terrain"]) == size * size


def test_snapshot_survives_the_jsonl_round_trip(tmp_path):
    """La restitution passe par le même stockage que le reste : si le format ne
    la transporte pas, la campagne archivée n'est pas rejouable."""
    result = run_scripted_match(seed=8, snapshots=True)
    path = write_game(tmp_path / "game.jsonl", result)
    reread = read_game(path)

    assert reread["world"] == result["world"]
    assert len(reread["history"]) == len(result["history"])
    for before, after in zip(result["history"], reread["history"]):
        assert after["board"] == before["board"]


def test_building_glyphs_are_pairwise_distinct():
    """Deux bâtiments ne peuvent pas partager un glyphe.

    « Camp militaire » et « Champs » commencent tous deux par C : dériver le
    glyphe du libellé les rendrait indiscernables à l'écran. Ce test tombe si
    un mod ajoute un bâtiment sans lui donner de glyphe propre.
    """
    catalog = build_catalog(base_config())
    glyphs = [building_glyph(key) for key in catalog.building_keys]
    assert len(set(glyphs)) == len(glyphs), (
        f"glyphes de bâtiment en collision : {dict(zip(catalog.building_keys, glyphs))}"
    )
    assert set(catalog.building_keys) <= set(BUILDING_GLYPHS), (
        "un bâtiment du catalogue n'a pas de glyphe déclaré"
    )


def test_world_snapshot_is_json_serialisable():
    """Un replay stocké doit pouvoir se relire seul, y compris par un navigateur."""
    cfg = base_config()
    catalog = build_catalog(cfg)
    from controla.rng import KeyedRNG
    from controla.state.game_state import create_state

    state = create_state(cfg, catalog, KeyedRNG(1))
    payload = world_snapshot(cfg, catalog, state)
    reread = json.loads(json.dumps(payload))
    assert reread["size"] == state.grid.size
    assert set(reread["glyphs"]) == set(catalog.building_keys)
    assert len(reread["factions"]) == int(cfg.world.num_factions)
