"""L'état de jeu : factions et état global, entièrement sérialisable.

Choix structurant : **la grille est la seule source de vérité du territoire.**
Une faction ne duplique pas la liste de ses cases ; on la dérive (`tiles_of`).
Cela supprime par construction toute une classe de bugs — l'incohérence entre le
bookkeeping d'une faction et la grille — et un invariant à tester avec.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from controla.registry.catalog import Catalog, TRAIT_ORDER
from controla.state.effects import EffectQueue, Modifier, active_multiplier, expire
from controla.world.grid import Coord, Grid

FACTION_NAMES = ("Cramoisi", "Cobalt", "Émeraude", "Ambre")
RESOURCE_KEYS = ("food", "wood", "metal")


@dataclass
class Reputation:
    """Compteur PUBLIC de promesses tenues et trahies (§7).

    C'est LE coût de la trahison, et il est volontairement **social** : pas de
    malus mécanique (un `+Guerrier` récompenserait le belliciste). Visible par
    tous les dieux, donc injecté dans toutes les observations.

    Le décompte porte sur l'**intention engagée**, pas sur l'issue : une alliance
    est honorée si le dieu a bien DÉPENSÉ l'action promise, gagnée ou perdue. Sans
    quoi le compteur mesurerait le dé et non la parole.
    """

    honored: int = 0
    betrayed: int = 0

    @property
    def total(self) -> int:
        return self.honored + self.betrayed

    @property
    def trust_rate(self) -> float | None:
        """Part de promesses tenues, ou None si le dieu ne s'est jamais engagé."""
        return self.honored / self.total if self.total else None

    def to_dict(self) -> dict:
        return {
            "honored": self.honored,
            "betrayed": self.betrayed,
            "total": self.total,
            "trust_rate": self.trust_rate,
        }


@dataclass
class Faction:
    """Un peuple. Les stocks sont des flottants : la passe travaille en continu et
    n'arrondit qu'à l'affichage (arrondir à chaque étape biaiserait l'économie)."""

    id: int
    name: str

    # Stocks
    food: float = 0.0
    wood: float = 0.0
    metal: float = 0.0

    # Stats accumulées
    population: float = 0.0
    research: float = 0.0
    research_tier: int = 0
    force: float = 0.0

    # Traits (§4) — 0.0 = baseline neutre, comme au départ pour les 4 factions
    traits: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in TRAIT_ORDER})
    # Ce que le peuple a vécu, lissé : c'est la « pression » de la dérive (§4)
    experience: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in TRAIT_ORDER})

    # Moyennes lissées (EMA) pilotant les envies de construction (§5)
    ema_production: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in RESOURCE_KEYS})
    ema_consumption: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in RESOURCE_KEYS})

    # Relations
    trade_relation: dict[int, float] = field(default_factory=dict)
    war_desire: dict[int, float] = field(default_factory=dict)
    at_war_with: set[int] = field(default_factory=set)

    # Réputation publique
    reputation: Reputation = field(default_factory=Reputation)

    # Budget du tour
    intervention_points: int = 0
    # Registre des actions payées ce tour : c'est LA preuve d'intention qui sert
    # à vérifier les commitments (§7, corrigé). Liste de (action_key, target).
    spent_this_turn: list[tuple[str, object]] = field(default_factory=list)

    # Faim du tour, dans [0, 1] : 0 = rassasié, 1 = rien à manger. Malus
    # MULTIPLICATIF sur la Force (§3), d'où une Force jamais négative.
    hunger: float = 0.0

    # Modificateurs datés portés par la faction (maladie sur la natalité, ...).
    # Même mécanique que sur une case : aucun cas particulier.
    modifiers: dict[str, Modifier] = field(default_factory=dict)

    # Brouillard
    revealed_until_turn: int = -1

    # Fin de partie
    eliminated_turn: int = -1

    # ── prédicats ────────────────────────────────────────────────────────────

    @property
    def alive(self) -> bool:
        return self.eliminated_turn < 0

    def stock(self, resource: str) -> float:
        return getattr(self, resource)

    def set_stock(self, resource: str, value: float) -> None:
        """Les stocks ne descendent jamais sous zéro — invariant du moteur."""
        setattr(self, resource, max(0.0, value))

    def modifier_multiplier(self, key: str, turn: int) -> float:
        """Multiplicateur d'un modificateur nommé, 1.0 s'il est absent ou échu."""
        modifier = self.modifiers.get(key)
        if modifier is None or modifier.until_turn < turn:
            return 1.0
        return max(0.0, 1.0 + modifier.amount)

    def expire_modifiers(self, turn: int) -> list[str]:
        return expire(self.modifiers, turn)

    def has_spent(self, action_key: str, target: object = None) -> bool:
        """A-t-il payé cette action ce tour ? Cible `None` = n'importe laquelle."""
        return any(
            key == action_key and (target is None or spent_target == target)
            for key, spent_target in self.spent_this_turn
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "food": round(self.food, 2),
            "wood": round(self.wood, 2),
            "metal": round(self.metal, 2),
            "population": round(self.population, 2),
            "research": round(self.research, 2),
            "research_tier": self.research_tier,
            "force": round(self.force, 2),
            "traits": {key: round(value, 4) for key, value in self.traits.items()},
            "experience": {key: round(value, 4) for key, value in self.experience.items()},
            "ema_production": {k: round(v, 3) for k, v in self.ema_production.items()},
            "ema_consumption": {k: round(v, 3) for k, v in self.ema_consumption.items()},
            "trade_relation": {str(k): round(v, 3) for k, v in self.trade_relation.items()},
            "war_desire": {str(k): round(v, 3) for k, v in self.war_desire.items()},
            "at_war_with": sorted(self.at_war_with),
            "reputation": self.reputation.to_dict(),
            "intervention_points": self.intervention_points,
            "spent_this_turn": [[key, target] for key, target in self.spent_this_turn],
            "hunger": round(self.hunger, 3),
            "modifiers": {key: mod.to_dict() for key, mod in self.modifiers.items()},
            "revealed_until_turn": self.revealed_until_turn,
            "eliminated_turn": self.eliminated_turn,
            "alive": self.alive,
        }


@dataclass
class GameState:
    """État complet d'une partie. `config`/`catalog`/`rng` sont des dépendances
    injectées, pas de l'état mutable de jeu."""

    turn: int
    grid: Grid
    factions: list[Faction]
    effects: EffectQueue = field(default_factory=EffectQueue)

    # Diplomatie (peuplée en tranche 3)
    negotiations: list = field(default_factory=list)
    commitments: list = field(default_factory=list)

    # ── territoire, dérivé de la grille ──────────────────────────────────────

    def tiles_of(self, faction_id: int) -> list:
        return self.grid.tiles_of(faction_id)

    def coords_of(self, faction_id: int) -> set[Coord]:
        return {tile.coord for tile in self.grid.tiles_of(faction_id)}

    def tile_count(self, faction_id: int) -> int:
        return sum(1 for tile in self.grid.all_tiles() if tile.owner == faction_id)

    def building_counts(self, faction_id: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tile in self.grid.all_tiles():
            if tile.owner == faction_id and tile.building is not None:
                counts[tile.building] = counts.get(tile.building, 0) + 1
        return counts

    def building_total(self, faction_id: int) -> int:
        return sum(self.building_counts(faction_id).values())

    # ── accès ────────────────────────────────────────────────────────────────

    @property
    def living(self) -> list[Faction]:
        return [faction for faction in self.factions if faction.alive]

    def faction(self, faction_id: int) -> Faction:
        return self.factions[faction_id]

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "grid": self.grid.to_dict(),
            "factions": [faction.to_dict() for faction in self.factions],
            "effects": self.effects.to_dict(),
        }


def create_state(cfg, catalog: Catalog, rng) -> GameState:
    """Monde symétrique, traits neutres, stocks identiques (§2)."""
    from controla.world.spawn import build_world

    grid, _regions = build_world(cfg, rng)
    count = int(cfg.world.num_factions)
    factions = [
        Faction(
            id=index,
            name=FACTION_NAMES[index] if index < len(FACTION_NAMES) else f"F{index}",
            food=float(cfg.start.food),
            wood=float(cfg.start.wood),
            metal=float(cfg.start.metal),
            population=float(cfg.start.population),
            research=float(cfg.start.research),
        )
        for index in range(count)
    ]
    for faction in factions:
        for other in factions:
            if other.id != faction.id:
                faction.trade_relation[other.id] = 0.0
                faction.war_desire[other.id] = 0.0
    return GameState(turn=0, grid=grid, factions=factions)
