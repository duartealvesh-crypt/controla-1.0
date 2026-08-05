"""Génération d'un monde à symétrie de rotation exacte (spec §2).

« Départs strictement symétriques → toute différence de résultat est attribuable
au **modèle**, pas au spawn. » On obtient cette garantie de la façon la plus forte
possible : la carte est invariante par rotation de 90°.

Concrètement, les cases sont regroupées en **orbites** sous la rotation
`(x, y) → (size-1-y, x)`. On tire le terrain une fois par orbite et on le recopie
sur les quatre cases. Les territoires des 4 factions étant eux-mêmes images l'un
de l'autre par cette rotation, ils ont des multi-ensembles de terrains
**identiques** — c'est prouvable, pas seulement équilibré en moyenne. Cela réduit
aussi la variance inter-parties, donc le nombre de parties nécessaires.
"""

from __future__ import annotations

from controla.world.grid import Coord, Grid


def rotate(x: int, y: int, size: int) -> Coord:
    """Rotation de 90° autour du centre de la grille. D'ordre 4 pour toute taille."""
    return (size - 1 - y, x)


def orbit(x: int, y: int, size: int) -> list[Coord]:
    """Les (jusqu'à 4) cases image l'une de l'autre par rotation.

    Une seule case est son propre point fixe : le centre exact d'une grille de
    taille impaire.
    """
    coords = [(x, y)]
    current = (x, y)
    for _ in range(3):
        current = rotate(*current, size)
        if current == (x, y):
            break
        coords.append(current)
    return coords


def _quotas(total: int, mix: dict[str, float]) -> dict[str, int]:
    """Répartit `total` cases selon `mix` par la méthode des plus forts restes.

    Les comptes tombent donc EXACTEMENT sur `total`, et les proportions réalisées
    sont celles configurées — pas des tirages indépendants qui varieraient d'une
    seed à l'autre (une source de variance en moins pour le benchmark).
    """
    weight_sum = sum(mix.values())
    if weight_sum <= 0:
        raise ValueError("world.terrain_mix : la somme des fréquences doit être > 0")
    exact = {key: total * weight / weight_sum for key, weight in mix.items()}
    counts = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(counts.values())
    # Ordre déterministe : reste décroissant, puis clé alphabétique.
    ranked = sorted(exact, key=lambda key: (-(exact[key] - counts[key]), key))
    for key in ranked[:remaining]:
        counts[key] += 1
    return counts


def generate_terrains(size: int, mix: dict[str, float], stream) -> list[list[str]]:
    """Grille de terrains invariante par rotation de 90°."""
    # Un représentant canonique par orbite, en ordre déterministe.
    seen: set[Coord] = set()
    canonical: list[Coord] = []
    for x in range(size):
        for y in range(size):
            if (x, y) in seen:
                continue
            group = orbit(x, y, size)
            seen.update(group)
            canonical.append((x, y))

    counts = _quotas(len(canonical), mix)
    pool: list[str] = []
    for terrain in sorted(counts):
        pool.extend([terrain] * counts[terrain])
    pool = stream.shuffled(pool)

    terrains = [[""] * size for _ in range(size)]
    for representative, terrain in zip(canonical, pool):
        for x, y in orbit(*representative, size):
            terrains[x][y] = terrain
    return terrains


def spawn_regions(size: int, corner_size: int) -> list[list[Coord]]:
    """Les territoires de départ des 4 factions, images l'un de l'autre.

    Faction 0 occupe le carré haut-gauche ; les suivantes sont ses rotations
    successives (donc, dans l'ordre, haut-droite, bas-droite, bas-gauche).
    """
    if corner_size * 2 > size:
        raise ValueError(
            f"spawn_corner_size={corner_size} trop grand pour une grille de {size} "
            "(les territoires de départ se chevaucheraient)"
        )
    region = [(x, y) for x in range(corner_size) for y in range(corner_size)]
    regions = [region]
    for _ in range(3):
        regions.append(sorted(rotate(x, y, size) for x, y in regions[-1]))
    return regions


def build_world(cfg, rng) -> tuple[Grid, list[list[Coord]]]:
    """Construit la grille et attribue les territoires de départ.

    L'aléa vient d'un flux keyé sur le tour 0 et le purpose « worldgen », donc la
    carte ne dépend que de la seed — pas du nombre d'actions jouées ensuite.
    """
    size = int(cfg.world.grid_size)
    stream = rng.stream(0, "worldgen")
    terrains = generate_terrains(size, dict(cfg.world.terrain_mix), stream)
    grid = Grid(size, terrains)

    regions = spawn_regions(size, int(cfg.world.spawn_corner_size))
    for faction_id, coords in enumerate(regions[: int(cfg.world.num_factions)]):
        for x, y in coords:
            grid.get(x, y).owner = faction_id
    return grid, regions
