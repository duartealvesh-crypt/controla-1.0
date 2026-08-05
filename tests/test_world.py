"""Le monde est strictement symétrique, et les distances sont correctes.

La symétrie n'est pas « équilibrée en moyenne » : elle est **exacte**, prouvée par
la rotation. C'est ce qui permet d'attribuer toute différence de résultat au modèle
et non au spawn (§2), et ce qui réduit la variance inter-parties.
"""

from __future__ import annotations

import pytest

from controla.config.loader import load_config
from controla.rng import KeyedRNG
from controla.world.grid import UNREACHABLE, Grid
from controla.world.spawn import (
    build_world,
    generate_terrains,
    orbit,
    rotate,
    spawn_regions,
)


def _grid_of(size: int, seed: int = 0, mix=None):
    stream = KeyedRNG(seed).stream(0, "worldgen")
    mix = mix or {"plain": 0.45, "river": 0.20, "forest": 0.35}
    return Grid(size, generate_terrains(size, mix, stream))


# ── rotation ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("size", [12, 13, 14, 15, 16])
def test_rotation_has_order_four(size):
    for x in range(size):
        for y in range(size):
            current = (x, y)
            for _ in range(4):
                current = rotate(*current, size)
            assert current == (x, y)


@pytest.mark.parametrize("size", [12, 13, 14, 15, 16])
def test_rotation_is_a_bijection(size):
    images = {rotate(x, y, size) for x in range(size) for y in range(size)}
    assert len(images) == size * size


def test_only_the_centre_of_an_odd_grid_is_a_fixed_point():
    assert len(orbit(6, 6, 13)) == 1          # centre exact de 13x13
    assert len(orbit(0, 0, 13)) == 4
    assert all(len(orbit(x, y, 14)) == 4 for x in range(14) for y in range(14))


# ── symétrie du terrain ──────────────────────────────────────────────────────

@pytest.mark.parametrize("size", [12, 13, 14, 15, 16])
@pytest.mark.parametrize("seed", [0, 7, 20260804])
def test_terrain_is_rotation_invariant(size, seed):
    grid = _grid_of(size, seed)
    for tile in grid.all_tiles():
        image = grid.get(*rotate(tile.x, tile.y, size))
        assert image.terrain == tile.terrain, (
            f"asymétrie en {tile.coord} -> {image.coord} "
            f"({tile.terrain} != {image.terrain})"
        )


def test_every_faction_starts_on_an_identical_terrain_multiset():
    """LE test qui garantit l'équité du spawn."""
    cfg = load_config()
    for seed in (0, 1, 42, 20260804):
        grid, _ = build_world(cfg, KeyedRNG(seed))
        multisets = []
        for faction_id in range(int(cfg.world.num_factions)):
            terrains = sorted(tile.terrain for tile in grid.tiles_of(faction_id))
            multisets.append(terrains)
        assert len(set(map(tuple, multisets))) == 1, (
            f"seed {seed} : les 4 départs n'ont pas le même terrain : {multisets}"
        )


def test_terrain_mix_hits_the_configured_proportions_exactly():
    """Quotas exacts plutôt que tirages indépendants : une variance en moins."""
    size, mix = 16, {"plain": 0.50, "river": 0.25, "forest": 0.25}
    grid = _grid_of(size, seed=3, mix=mix)
    counts: dict[str, int] = {}
    for tile in grid.all_tiles():
        counts[tile.terrain] = counts.get(tile.terrain, 0) + 1
    total = size * size
    for terrain, share in mix.items():
        assert counts[terrain] == pytest.approx(total * share, abs=4)


def test_terrain_generation_is_seed_deterministic():
    a = [tile.terrain for tile in _grid_of(14, seed=5).all_tiles()]
    b = [tile.terrain for tile in _grid_of(14, seed=5).all_tiles()]
    c = [tile.terrain for tile in _grid_of(14, seed=6).all_tiles()]
    assert a == b
    assert a != c


# ── territoires de départ ────────────────────────────────────────────────────

def test_spawn_regions_are_disjoint_and_equal_sized():
    regions = spawn_regions(14, 3)
    assert len(regions) == 4
    assert all(len(region) == 9 for region in regions)
    flat = [coord for region in regions for coord in region]
    assert len(flat) == len(set(flat)), "des territoires de départ se chevauchent"


def test_spawn_regions_sit_in_the_four_corners():
    """Chaque coin de la grille appartient à exactement une faction."""
    regions = spawn_regions(14, 3)
    for corner in ((0, 0), (13, 0), (13, 13), (0, 13)):
        owners = [index for index, region in enumerate(regions) if corner in region]
        assert len(owners) == 1, f"le coin {corner} appartient à {owners}"


def test_spawn_rejects_an_oversized_corner():
    with pytest.raises(ValueError, match="trop grand"):
        spawn_regions(10, 6)


def test_build_world_assigns_exactly_the_starting_territories():
    cfg = load_config()
    grid, regions = build_world(cfg, KeyedRNG(11))
    expected = int(cfg.world.spawn_corner_size) ** 2
    for faction_id in range(int(cfg.world.num_factions)):
        assert len(grid.tiles_of(faction_id)) == expected
        assert {tile.coord for tile in grid.tiles_of(faction_id)} == set(regions[faction_id])
    owned = sum(1 for tile in grid.all_tiles() if tile.owner is not None)
    assert owned == expected * int(cfg.world.num_factions)


def test_starting_positions_are_mutually_distant():
    """Petite carte volontairement : les contacts doivent arriver tôt, mais pas
    au tour 1 (§2)."""
    cfg = load_config()
    grid, _ = build_world(cfg, KeyedRNG(0))
    assert not grid.contiguous_pairs(), "des factions se touchent dès le départ"


# ── grille et distances ──────────────────────────────────────────────────────

def test_bfs_distance_is_manhattan_on_an_empty_grid():
    grid = _grid_of(14)
    distances = grid.bfs_distance_map({(0, 0)})
    assert distances[0][0] == 0
    assert distances[3][4] == 7
    assert distances[13][13] == 26


def test_bfs_with_no_source_leaves_everything_unreachable():
    grid = _grid_of(8)
    distances = grid.bfs_distance_map(set())
    assert all(cell == UNREACHABLE for row in distances for cell in row)


def test_bfs_takes_the_nearest_of_several_sources():
    grid = _grid_of(10)
    distances = grid.bfs_distance_map({(0, 0), (9, 9)})
    assert distances[8][9] == 1
    assert distances[5][5] == 8       # 10 depuis (0,0), 8 depuis (9,9)
    assert distances[2][2] == 4       # 4 depuis (0,0), 14 depuis (9,9)


def test_neighbours_are_clipped_at_the_border():
    grid = _grid_of(6)
    assert len(grid.neighbours(0, 0)) == 2
    assert len(grid.neighbours(0, 3)) == 3
    assert len(grid.neighbours(3, 3)) == 4


def test_contiguous_pairs_detects_adjacency_only():
    grid = _grid_of(6)
    grid.get(2, 2).owner = 0
    grid.get(2, 3).owner = 1
    grid.get(5, 5).owner = 2
    assert grid.contiguous_pairs() == {(0, 1)}


def test_empty_neighbours_of_is_deduplicated_and_deterministic():
    grid = _grid_of(6)
    grid.get(1, 1).owner = 0
    grid.get(1, 2).owner = 0
    coords = [tile.coord for tile in grid.empty_neighbours_of({(1, 1), (1, 2)})]

    assert len(coords) == len(set(coords)), "le front d'expansion contient un doublon"
    assert not ({(1, 1), (1, 2)} & set(coords)), "une case possédée est dans le front"
    assert set(coords) == {(0, 1), (2, 1), (1, 0), (0, 2), (2, 2), (1, 3)}
    # L'ordre doit être reproductible : il pilote un tirage dans la passe.
    assert coords == [t.coord for t in grid.empty_neighbours_of({(1, 2), (1, 1)})]


# ── modificateurs de case ────────────────────────────────────────────────────

def test_tile_modifiers_stack_and_expire():
    from controla.state.effects import Modifier

    grid = _grid_of(6)
    tile = grid.get(0, 0)
    tile.modifiers["blessing"] = Modifier(0.5, until_turn=3)
    tile.modifiers["disaster"] = Modifier(-0.2, until_turn=1)

    assert tile.modifier_multiplier(1) == pytest.approx(1.5 * 0.8)
    assert tile.modifier_multiplier(2) == pytest.approx(1.5)
    assert tile.modifier_multiplier(4) == pytest.approx(1.0)

    assert tile.expire_modifiers(2) == ["disaster"]
    assert set(tile.modifiers) == {"blessing"}


def test_tile_productivity_never_goes_negative():
    from controla.state.effects import Modifier

    grid = _grid_of(6)
    tile = grid.get(0, 0)
    tile.modifiers["a"] = Modifier(-2.0, until_turn=5)
    assert tile.modifier_multiplier(1) == 0.0
