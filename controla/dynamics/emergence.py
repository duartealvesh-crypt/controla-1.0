"""Comportements émergents : expansion, constructions, commerce, guerre (spec §5, §8.9).

C'est ici que vit le principe le plus important du jeu : **le peuple est
non-agentique**. Il n'y a ni autopilote ni seconde IA. Ce qui ressemble à « le
peuple agit » n'est que ces règles plus l'aléa de la passe.

Mais *indirect ≠ aléatoire* : les règles driver → comportement sont déterministes
et **lisibles**. Le dieu doit pouvoir prédire l'effet de son influence. On cache les
coefficients, jamais les causes.
"""

from __future__ import annotations

from controla.registry.catalog import Catalog
from controla.registry.descriptors import BuildingSpec
from controla.state.game_state import RESOURCE_KEYS, Faction, GameState


# ── valeurs de driver ────────────────────────────────────────────────────────

def driver_value(state: GameState, faction: Faction, key: str) -> float:
    """Valeur d'un driver d'envie : clé de trait ou clé de stat.

    Un seul point de résolution, pour qu'ajouter un driver dans le YAML ne demande
    aucun code (§11).
    """
    if key in faction.traits:
        return faction.traits[key]
    if key == "population":
        return faction.population
    if key == "tiles":
        return float(state.tile_count(faction.id))
    if key == "building_density":
        tiles = state.tile_count(faction.id)
        return 0.0 if tiles == 0 else state.building_total(faction.id) / tiles
    raise KeyError(f"driver d'envie inconnu : {key!r}")


def relative_deficit(faction: Faction, resource: str | None) -> float:
    """`D = max(0, (C̄ − P̄)/C̄)` sur les moyennes lissées.

    Sur des moyennes et non sur l'instant : la passe est stochastique, et sans
    lissage l'envie spikerait à chaque mauvaise récolte → sur-construction.
    """
    if resource is None:
        return 0.0
    smoothed_need = faction.ema_consumption.get(resource, 0.0)
    if smoothed_need <= 0:
        return 0.0
    smoothed_output = faction.ema_production.get(resource, 0.0)
    return max(0.0, (smoothed_need - smoothed_output) / smoothed_need)


def saturation(count: int, halflife: float | None) -> float:
    """`s(count)` : rendements décroissants sur le nombre déjà construit.

    Nécessaire **seulement** pour les bâtiments sans boucle de déficit (camp, labo).
    Pour les champs elle est redondante — bâtir fait monter `P̄`, donc baisser `D`,
    donc l'envie retombe d'elle-même — d'où `saturation_halflife: null`.
    """
    if halflife is None or halflife <= 0:
        return 1.0
    return halflife / (count + halflife)


def building_desire(cfg, state: GameState, faction: Faction, spec: BuildingSpec) -> float:
    """L'envie de construire ce bâtiment, dans l'esprit de §5.

    `e_base` est un plancher **additif** : même sans déficit, un peuple construit.
    `criticality_k ≤ 1` fait réagir tôt sur une ressource vitale (bâtir des champs
    avant la famine, pas après) ; `k ≥ 2` réserve le luxe à l'abondance.
    """
    deficit = relative_deficit(faction, spec.deficit_resource)
    urgency = spec.e_base + (1.0 - spec.e_base) * (deficit ** spec.criticality_k)
    drivers = sum(
        weight * driver_value(state, faction, key)
        for key, weight in spec.drivers.items()
    )
    count = state.building_counts(faction.id).get(spec.key, 0)
    return urgency * max(0.0, 1.0 + drivers) * saturation(count, spec.saturation_halflife)


# ── choix de case ────────────────────────────────────────────────────────────

def _tile_score(catalog: Catalog, tile, resource: str | None) -> float:
    """Qualité d'une case pour un usage donné.

    Une seule règle, pas de cas par bâtiment : on note sur la ressource visée si le
    bâtiment répond à un déficit (les champs vont sur la nourriture), sinon sur le
    rendement total.
    """
    terrain = next(spec for spec in catalog.terrains if spec.key == tile.terrain)
    if resource is not None:
        return terrain.produces.get(resource, 0.0)
    return sum(terrain.produces.values())


def _pick(stream, cfg, options: list, scores: list[float]):
    """Arbitrage stochastique biaisé vers le meilleur — jamais un argmax pur, sinon
    deux peuples identiques joueraient exactement pareil."""
    return stream.softmax_choice(options, scores, float(cfg.emergence.softmax_temperature))


# ── 9a. Expansion ───────────────────────────────────────────────────────────

def expand(cfg, catalog: Catalog, state: GameState, rng) -> list[dict]:
    """Le peuple s'étend seul vers les cases vides adjacentes."""
    events: list[dict] = []
    for faction in state.living:
        stream = rng.stream(state.turn, "expansion", faction.id)
        desire = (
            float(cfg.emergence.expansion_pop_weight) * faction.population
            + float(cfg.emergence.expansion_density_weight)
            * driver_value(state, faction, "building_density")
        )
        threshold = stream.noisy(
            float(cfg.emergence.expansion_threshold), float(cfg.emergence.threshold_sigma)
        )
        if desire <= threshold:
            continue

        frontier = state.grid.empty_neighbours_of(state.coords_of(faction.id))
        if not frontier:
            continue
        scores = [_tile_score(catalog, tile, None) for tile in frontier]
        tile = _pick(stream, cfg, frontier, scores)
        tile.owner = faction.id
        events.append({
            "kind": "expansion", "faction": faction.id, "tile": list(tile.coord),
            "cause": f"envie d'expansion {desire:.2f} > seuil {threshold:.2f}",
        })
    return events


# ── 9b. Constructions ───────────────────────────────────────────────────────

def build(cfg, catalog: Catalog, state: GameState, rng) -> list[dict]:
    """Au plus une construction par faction et par tour, choisie par softmax.

    Aucun bâtiment n'est ordonné : chacun émerge quand son envie franchit un seuil
    bruité. Le bois disponible plafonne naturellement le rythme.
    """
    events: list[dict] = []
    for faction in state.living:
        stream = rng.stream(state.turn, "build", faction.id)
        free_tiles = [
            tile for tile in state.tiles_of(faction.id) if tile.building is None
        ]
        if not free_tiles:
            continue

        candidates: list[BuildingSpec] = []
        desires: list[float] = []
        for spec in catalog.buildings:
            if faction.wood < spec.wood_cost:
                continue
            desire = building_desire(cfg, state, faction, spec)
            threshold = stream.noisy(spec.threshold, float(cfg.emergence.threshold_sigma))
            if desire > threshold:
                candidates.append(spec)
                desires.append(desire)
        if not candidates:
            continue

        spec = _pick(stream, cfg, candidates, desires)
        scores = [_tile_score(catalog, tile, spec.deficit_resource) for tile in free_tiles]
        tile = _pick(stream, cfg, free_tiles, scores)

        faction.set_stock("wood", faction.wood - spec.wood_cost)
        tile.building = spec.key
        events.append({
            "kind": "build", "faction": faction.id, "building": spec.key,
            "tile": list(tile.coord),
            "cause": f"envie de {spec.label} {desires[candidates.index(spec)]:.2f}",
        })
    return events


# ── 9c. Commerce ────────────────────────────────────────────────────────────

def set_trade_relation(state: GameState, a: int, b: int, value: float) -> None:
    """La relation commerciale est **symétrique** : un embargo d'un côté arrête les
    échanges des deux côtés (§6). On la garde donc synchronisée par construction."""
    clamped = max(-1.0, min(1.0, value))
    state.faction(a).trade_relation[b] = clamped
    state.faction(b).trade_relation[a] = clamped


def trade(cfg, state: GameState, rng) -> list[dict]:
    """Échanges automatiques au-delà du seuil : chacun donne son abondance et reçoit
    son manque.

    Les transferts **conservent** les ressources : ce qui part d'un stock arrive
    dans l'autre, à l'unité près. Un invariant de test le vérifie.
    """
    events: list[dict] = []
    decay = float(cfg.trade.decay)
    living = state.living

    for index, first in enumerate(living):
        for second in living[index + 1:]:
            current = first.trade_relation.get(second.id, 0.0)
            set_trade_relation(state, first.id, second.id, current * decay)

    for index, first in enumerate(living):
        for second in living[index + 1:]:
            relation = first.trade_relation.get(second.id, 0.0)
            if relation < float(cfg.trade.threshold):
                continue
            moved: dict[str, float] = {}
            for resource in RESOURCE_KEYS:
                gap = first.stock(resource) - second.stock(resource)
                if abs(gap) < 1e-9:
                    continue
                amount = float(cfg.trade.fraction) * abs(gap) / 2.0
                giver, taker = (first, second) if gap > 0 else (second, first)
                giver.set_stock(resource, giver.stock(resource) - amount)
                taker.set_stock(resource, taker.stock(resource) + amount)
                moved[resource] = round(amount, 3)
            if moved:
                events.append({
                    "kind": "trade", "factions": [first.id, second.id],
                    "moved": moved,
                    "cause": f"relation commerciale {relation:.2f} au-dessus du seuil",
                })
    return events


# ── 9d. Déclenchement de guerre ─────────────────────────────────────────────

def _attackable_tiles(state: GameState, attacker_id: int, defender_id: int) -> list:
    """Cases de la cible adjacentes à notre territoire — la portée d'une attaque."""
    own = state.coords_of(attacker_id)
    out = []
    for tile in state.tiles_of(defender_id):
        if any(
            neighbour.coord in own
            for neighbour in state.grid.neighbours(tile.x, tile.y)
        ):
            out.append(tile)
    return out


def detects_attack(state: GameState, defender: Faction, attacker_id: int) -> bool:
    """Le défenseur voit-il l'attaque partir ?

    Les armées en transit sont cachées par le brouillard. Il ne la voit donc que
    s'il est déjà en guerre avec cet attaquant (il le surveille) ou s'il a
    l'Omniscience active. C'est ce qui fait de la surprise la récompense d'une
    traîtrise réussie, plutôt qu'un bonus permanent.
    """
    return attacker_id in defender.at_war_with or defender.revealed_until_turn >= state.turn


def trigger_wars(cfg, catalog: Catalog, state: GameState, rng) -> list[dict]:
    """Le peuple déclenche la guerre quand son envie franchit le seuil.

    Le dieu ne décrète jamais une attaque : son *Appel à la guerre* pousse l'envie
    (proportionnellement au trait Guerrier), et le peuple décide. L'attaque part
    maintenant et se résout dans `march_turns` tours — c'est la fenêtre de réaction
    du défenseur, s'il la voit venir.
    """
    events: list[dict] = []
    decay = float(cfg.emergence.war_decay)

    for faction in state.living:
        stream = rng.stream(state.turn, "war_trigger", faction.id)
        for target_id in sorted(faction.war_desire):
            faction.war_desire[target_id] *= decay

        for target_id in sorted(faction.war_desire):
            defender = state.faction(target_id)
            if not defender.alive:
                continue
            appetite = faction.war_desire[target_id] * max(
                0.0, 1.0 + faction.traits["warrior"]
            )
            threshold = stream.noisy(
                float(cfg.emergence.war_threshold), float(cfg.emergence.threshold_sigma)
            )
            if appetite <= threshold:
                continue

            reachable = _attackable_tiles(state, faction.id, target_id)
            if not reachable:
                continue
            scores = [
                _tile_score(catalog, tile, None) + (2.0 if tile.building else 0.0)
                for tile in reachable
            ]
            tile = _pick(stream, cfg, reachable, scores)

            # La surprise est figée AU DÉPART, quand la détection a lieu — pas à la
            # résolution, où le défenseur sera de toute façon en guerre.
            surprise = not detects_attack(state, defender, faction.id)
            state.effects.add(
                "attack",
                due_turn=state.turn + int(cfg.combat.march_turns),
                actor=faction.id,
                cause=f"envie de guerre {appetite:.2f} > seuil {threshold:.2f}",
                defender=target_id,
                tile=tile.coord,
                surprise=surprise,
            )
            faction.at_war_with.add(target_id)
            defender.at_war_with.add(faction.id)
            faction.war_desire[target_id] = 0.0   # l'envie est consommée par l'assaut
            events.append({
                "kind": "war_launched", "faction": faction.id, "target": target_id,
                "tile": list(tile.coord), "surprise": surprise,
                "cause": f"envie de guerre {appetite:.2f} > seuil {threshold:.2f}",
            })
    return events
