"""La grille et ses cases.

Une case porte son terrain (immuable), son propriétaire, au plus un bâtiment, et
un dictionnaire de modificateurs datés (bénédiction, désastre, événement). Les
modificateurs sont génériques : aucun cas particulier par miracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from controla.state.effects import Modifier, active_multiplier, expire

Coord = tuple[int, int]

# Voisinage à 4 : c'est lui qui définit l'adjacence pour l'expansion, la
# contiguïté des factions et les distances BFS.
NEIGHBOUR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))

UNREACHABLE = -1


@dataclass
class Tile:
    x: int
    y: int
    terrain: str
    owner: int | None = None
    building: str | None = None
    modifiers: dict[str, Modifier] = field(default_factory=dict)

    @property
    def coord(self) -> Coord:
        return (self.x, self.y)

    def modifier_multiplier(self, turn: int) -> float:
        return active_multiplier(self.modifiers, turn)

    def expire_modifiers(self, turn: int) -> list[str]:
        return expire(self.modifiers, turn)

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "terrain": self.terrain,
            "owner": self.owner,
            "building": self.building,
            "modifiers": {key: mod.to_dict() for key, mod in self.modifiers.items()},
        }


class Grid:
    """Grille carrée de `size`×`size` cases."""

    __slots__ = ("size", "_tiles")

    def __init__(self, size: int, terrains: list[list[str]]):
        self.size = size
        self._tiles: list[list[Tile]] = [
            [Tile(x, y, terrains[x][y]) for y in range(size)] for x in range(size)
        ]

    # ── accès ────────────────────────────────────────────────────────────────

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def get(self, x: int, y: int) -> Tile:
        return self._tiles[x][y]

    def all_tiles(self):
        for column in self._tiles:
            yield from column

    def neighbours(self, x: int, y: int) -> list[Tile]:
        return [
            self._tiles[x + dx][y + dy]
            for dx, dy in NEIGHBOUR_OFFSETS
            if self.in_bounds(x + dx, y + dy)
        ]

    def tiles_of(self, owner: int) -> list[Tile]:
        """Cases d'un propriétaire, en ordre déterministe (x puis y)."""
        return [tile for tile in self.all_tiles() if tile.owner == owner]

    def empty_neighbours_of(self, coords: set[Coord]) -> list[Tile]:
        """Cases vides adjacentes à un ensemble de cases, sans doublon et en ordre
        déterministe — c'est le front d'expansion."""
        seen: set[Coord] = set()
        out: list[Tile] = []
        for x, y in sorted(coords):
            for tile in self.neighbours(x, y):
                if tile.owner is None and tile.coord not in seen:
                    seen.add(tile.coord)
                    out.append(tile)
        return out

    def contiguous_pairs(self) -> set[tuple[int, int]]:
        """Paires de factions dont les territoires se touchent (ordonnées)."""
        pairs: set[tuple[int, int]] = set()
        for tile in self.all_tiles():
            if tile.owner is None:
                continue
            for neighbour in self.neighbours(tile.x, tile.y):
                if neighbour.owner is not None and neighbour.owner != tile.owner:
                    pairs.add(tuple(sorted((tile.owner, neighbour.owner))))  # type: ignore[arg-type]
        return pairs

    # ── distances ────────────────────────────────────────────────────────────

    def bfs_distance_map(self, sources: set[Coord]) -> list[list[int]]:
        """Distance à 4-voisinage vers la source la plus proche (0 sur une source).

        Cases inatteignables à `UNREACHABLE` ; sur une grille pleine cela n'arrive
        que si `sources` est vide. File en liste + index de tête : pas besoin de
        `deque`, et l'ordre de parcours reste déterministe.
        """
        distances = [[UNREACHABLE] * self.size for _ in range(self.size)]
        frontier: list[Coord] = []
        for x, y in sorted(sources):
            if self.in_bounds(x, y):
                distances[x][y] = 0
                frontier.append((x, y))
        head = 0
        while head < len(frontier):
            x, y = frontier[head]
            head += 1
            next_distance = distances[x][y] + 1
            for dx, dy in NEIGHBOUR_OFFSETS:
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny) and distances[nx][ny] == UNREACHABLE:
                    distances[nx][ny] = next_distance
                    frontier.append((nx, ny))
        return distances

    # ── sérialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "tiles": [tile.to_dict() for tile in self.all_tiles()],
        }
