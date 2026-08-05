"""Étapes économiques de la passe : production, consommation, population, recherche.

Chaque fonction est une étape de la passe §8, dans l'ordre. Elles renvoient un petit
dict « ce qui s'est passé » pour l'**attribution** : le dieu doit voir sa causalité
même médiée (« Bénédiction (rivière nord) : +4.2 nourriture »).

Les stocks sont des flottants du début à la fin. Arrondir à chaque étape biaiserait
l'économie de façon systématique (toujours dans le même sens) ; on n'arrondit qu'à
l'affichage.
"""

from __future__ import annotations

from controla.registry.catalog import Catalog
from controla.state.game_state import RESOURCE_KEYS, Faction, GameState


def trait_multiplier(faction: Faction, trait: str, weight: float) -> float:
    """`1 + poids × trait`, borné à 0.

    Avec des traits dans [-50 %, +150 %] et des poids raisonnables, le résultat est
    toujours positif ; la borne protège d'une config de sweep extrême plutôt que
    de laisser apparaître une production négative.
    """
    return max(0.0, 1.0 + weight * faction.traits[trait])


# ── 2. Production ────────────────────────────────────────────────────────────

def tile_yield(
    cfg, catalog: Catalog, state: GameState, faction: Faction, tile, stream
) -> dict[str, float]:
    """Rendement d'une case, tous modificateurs appliqués.

    Ordre : terrain → bâtiment → modificateurs datés → trait Travailleur →
    palier de recherche → aléa. Le bruit est multiplicatif et borné à 0 : une case
    ne produit jamais négativement.
    """
    terrain = next(spec for spec in catalog.terrains if spec.key == tile.terrain)
    worker = trait_multiplier(faction, "worker", float(cfg.production.worker_weight))
    modifiers = tile.modifier_multiplier(state.turn)

    out: dict[str, float] = {}
    for resource, base in terrain.produces.items():
        if base <= 0:
            continue
        amount = base
        if tile.building == "field" and resource == "food":
            amount *= float(cfg.production.field_food_bonus)
        if tile.building == "city":
            amount *= float(cfg.production.city_multiplier)
        amount *= modifiers * worker
        if resource == "food" and faction.research_tier >= 2:
            amount *= 1.0 + float(cfg.research.tier2_food_bonus)
        amount = max(0.0, stream.noisy(amount, amount * float(cfg.production.noise_sigma)))
        out[resource] = out.get(resource, 0.0) + amount
    return out


def produce(cfg, catalog: Catalog, state: GameState, rng) -> dict[int, dict]:
    """Production de toutes les factions vivantes.

    Le métal ne vient d'aucun terrain (§2) : il est débloqué par le palier de
    recherche 1 et produit proportionnellement au nombre de cases.
    """
    report: dict[int, dict] = {}
    for faction in state.living:
        stream = rng.stream(state.turn, "production", faction.id)
        totals = {key: 0.0 for key in RESOURCE_KEYS}
        for tile in state.tiles_of(faction.id):
            for resource, amount in tile_yield(
                cfg, catalog, state, faction, tile, stream
            ).items():
                totals[resource] += amount

        if faction.research_tier >= 1:
            totals["metal"] += float(cfg.production.metal_per_tile) * state.tile_count(faction.id)

        for resource, amount in totals.items():
            faction.set_stock(resource, faction.stock(resource) + amount)
        report[faction.id] = {"produced": totals}
    return report


# ── 3. Consommation ─────────────────────────────────────────────────────────

def consume(cfg, state: GameState) -> dict[int, dict]:
    """Consommation de nourriture et calcul de la faim.

    `hunger ∈ [0, 1]` mesure la part du besoin non couverte. C'est l'« envie de
    nourriture » de §3 : un malus **multiplicatif** sur la Force, donc la Force ne
    peut jamais devenir négative, quelle que soit la famine.
    """
    report: dict[int, dict] = {}
    for faction in state.living:
        per_capita = float(cfg.consumption.food_per_capita) * max(
            0.0,
            1.0 - float(cfg.consumption.resilient_weight) * faction.traits["resilient"],
        )
        if faction.research_tier >= 3:
            per_capita *= 1.0 - float(cfg.research.tier3_consumption_cut)

        need = per_capita * faction.population
        eaten = min(faction.food, need)
        faction.set_stock("food", faction.food - eaten)
        faction.hunger = 0.0 if need <= 0 else max(0.0, min(1.0, 1.0 - eaten / need))
        report[faction.id] = {"need": need, "eaten": eaten, "hunger": faction.hunger}
    return report


def update_smoothing(cfg, state: GameState, production: dict[int, dict],
                     consumption: dict[int, dict]) -> None:
    """Met à jour les moyennes lissées (EMA) qui pilotent les envies (§5).

    Sans ce lissage, la passe stochastique fait spiker l'envie et le peuple
    sur-construit. `alpha` est LE bouton réactif ↔ stable.

    La consommation n'est lissée que pour la nourriture ; le bois et le métal n'ont
    pas de consommation continue (ils se dépensent par à-coups), donc leur
    « déficit » n'aurait pas de sens.
    """
    alpha = float(cfg.emergence.ema_alpha)
    for faction in state.living:
        produced = production.get(faction.id, {}).get("produced", {})
        for resource in RESOURCE_KEYS:
            current = float(produced.get(resource, 0.0))
            faction.ema_production[resource] = (
                alpha * current + (1.0 - alpha) * faction.ema_production[resource]
            )
        need = float(consumption.get(faction.id, {}).get("need", 0.0))
        faction.ema_consumption["food"] = (
            alpha * need + (1.0 - alpha) * faction.ema_consumption["food"]
        )


# ── 4. Population (croissance logistique) ───────────────────────────────────

def carrying_capacity(cfg, state: GameState, faction: Faction) -> float:
    """`K` : le plafond de population.

    Il dépend de la production de nourriture **lissée** (pas du stock : un stock
    ponctuel ne nourrit pas durablement) et du nombre de villes. Toujours > 0 pour
    que la logistique reste définie.
    """
    cities = state.building_counts(faction.id).get("city", 0)
    return max(
        1.0,
        float(cfg.population.carrying_base)
        + float(cfg.population.carrying_per_food) * faction.ema_production["food"]
        + float(cfg.population.carrying_per_city) * cities,
    )


def grow_population(cfg, state: GameState) -> dict[int, dict]:
    """`P += r·P·(1 − P/K)`, puis pertes de famine.

    La population croît vite loin du plafond, se stabilise à `K` et **décline
    au-dessus** — c'est exactement « la population monte jusqu'à ce que la
    nourriture la limite ». La famine s'ajoute par-dessus, seule voie vers
    l'élimination par épuisement.
    """
    report: dict[int, dict] = {}
    for faction in state.living:
        capacity = carrying_capacity(cfg, state, faction)
        rate = (
            float(cfg.population.growth_rate)
            * trait_multiplier(faction, "fertility", float(cfg.population.fertility_weight))
            * faction.modifier_multiplier("natality", state.turn)
        )
        before = faction.population
        logistic = rate * before * (1.0 - before / capacity)
        starved = float(cfg.population.starvation_death_rate) * faction.hunger * before
        faction.population = max(0.0, before + logistic - starved)
        report[faction.id] = {
            "capacity": capacity,
            "logistic": logistic,
            "starved": starved,
            "population": faction.population,
        }
    return report


# ── 5. Recherche ────────────────────────────────────────────────────────────

def advance_research(cfg, state: GameState) -> dict[int, dict]:
    """Progression de la Recherche et franchissement des paliers.

    Les paliers sont cumulatifs et monotones : une fois atteint, un palier n'est
    jamais reperdu (perdre le Métal en cours de partie serait illisible pour le dieu).
    """
    thresholds = list(cfg.research.tiers)
    report: dict[int, dict] = {}
    for faction in state.living:
        labs = state.building_counts(faction.id).get("lab", 0)
        gain = (
            float(cfg.research.base_rate)
            + float(cfg.research.per_lab) * labs
        ) * trait_multiplier(
            faction, "intelligence", float(cfg.research.intelligence_weight)
        )
        faction.research += gain

        reached = sum(1 for threshold in thresholds if faction.research >= threshold)
        unlocked = reached > faction.research_tier
        faction.research_tier = max(faction.research_tier, reached)
        report[faction.id] = {
            "gain": gain,
            "research": faction.research,
            "tier": faction.research_tier,
            "unlocked": unlocked,
        }
    return report


# ── 7. Points d'intervention ────────────────────────────────────────────────

def recharge_intervention_points(cfg, state: GameState) -> dict[int, int]:
    """Le pouvoir du dieu croît avec son peuple, mais plafonne.

    `log(1 + population)` plutôt que linéaire, et un plafond dur : c'est la
    correction anti-snowball du plan. Un scaling linéaire ou exponentiel ferait du
    leader un dieu deux fois plus puissant, ce qui *crée* la stratégie dominante
    que §12 interdit.
    """
    import math

    report: dict[int, int] = {}
    for faction in state.living:
        raw = (
            float(cfg.intervention.base)
            + float(cfg.intervention.pop_weight) * math.log1p(max(0.0, faction.population))
            + float(cfg.intervention.tile_weight) * state.tile_count(faction.id)
            + float(cfg.intervention.city_weight)
            * state.building_counts(faction.id).get("city", 0)
        )
        faction.intervention_points = min(int(cfg.intervention.cap), int(raw))
        report[faction.id] = faction.intervention_points
    return report
