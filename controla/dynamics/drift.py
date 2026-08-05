"""Dérive des traits sous pression : l'héritage émergent (spec §4).

```
trait += learning_rate × (pression − trait) + petit_bruit
```

La **pression** vient de ce que le peuple a vécu cette ère. C'est la version
émergente du tempérament : le dieu ne *choisit* pas ce que devient son peuple, ses
décisions en sont la **pression de sélection**. Un seul levier, deux échelles de
temps — le comportement immédiat (l'Appel) et l'héritage long (la dérive).

Le vécu est lui-même lissé (`faction.experience`) : une seule guerre ne fait pas un
peuple guerrier, une série de guerres si.
"""

from __future__ import annotations

from controla.state.game_state import Faction, GameState

# Ce qui met chaque trait sous pression. Les sources sont déclarées ici, en données ;
# `measure_pressure` les traduit en un signal dans [0, 1]. Les libellés lisibles
# vivent dans le catalogue (`TraitSpec.pressure_sources`), pas ici.
PRESSURE_SIGNALS: dict[str, tuple[str, ...]] = {
    "builder": ("built",),
    "warrior": ("fought", "attacked"),
    "fertility": ("grew",),
    "resilient": ("starved", "struck"),
    "intelligence": ("labs",),
    "worker": ("terrain_wealth", "blessed"),
}


def measure_pressure(cfg, state: GameState, faction: Faction, signals: dict) -> dict[str, float]:
    """Traduit le vécu du tour en un signal de pression par trait, dans [0, 1].

    Chaque signal est normalisé pour que la pression soit comparable à un trait
    (qui vit dans [`traits.min`, `traits.max`]) : on la projette ensuite sur cette
    échelle dans `apply_drift`.
    """
    tiles = max(1, state.tile_count(faction.id))
    buildings = state.building_counts(faction.id)
    raw = {
        "built": 1.0 if signals.get("built") else 0.0,
        "fought": 1.0 if signals.get("fought") else 0.0,
        "attacked": 1.0 if signals.get("attacked") else 0.0,
        "grew": 1.0 if signals.get("grew", 0.0) > 0 else 0.0,
        "starved": faction.hunger,
        "struck": 1.0 if signals.get("struck") else 0.0,
        "labs": min(1.0, buildings.get("lab", 0) / 3.0),
        "terrain_wealth": min(1.0, faction.ema_production["food"] / max(1.0, tiles * 2.0)),
        "blessed": 1.0 if signals.get("blessed") else 0.0,
    }
    return {
        trait: max(raw[name] for name in names)
        for trait, names in PRESSURE_SIGNALS.items()
    }


def apply_drift(
    cfg, state: GameState, rng, signals_by_faction: dict[int, dict]
) -> list[dict]:
    """Fait dériver chaque trait vers sa pression, en restant dans les bornes."""
    low, high = float(cfg.traits.min), float(cfg.traits.max)
    rate = float(cfg.traits.drift_rate)
    sigma = float(cfg.traits.drift_noise_sigma)
    alpha = float(cfg.emergence.ema_alpha)
    events: list[dict] = []

    for faction in state.living:
        stream = rng.stream(state.turn, "drift", faction.id)
        pressure = measure_pressure(
            cfg, state, faction, signals_by_faction.get(faction.id, {})
        )
        moved: dict[str, float] = {}

        for trait in sorted(pressure):
            # La pression est un signal dans [0, 1], projeté sur [0, borne haute] —
            # et NON sur [borne basse, borne haute].
            #
            # Le neutre (0) est la baseline : ne pas faire la guerre doit ramener un
            # peuple vers l'ordinaire, pas le rendre pacifiste à -50 %. Mapper sur
            # la borne basse ferait dériver TOUS les traits non sollicités vers leur
            # minimum, ce qui viderait le jeu de sa spécialisation. Seuls les Appels
            # du dieu peuvent pousser un trait sous le neutre.
            target = high * pressure[trait]
            # Le vécu est lissé : une guerre isolée ne fait pas un peuple guerrier.
            faction.experience[trait] = (
                alpha * target + (1.0 - alpha) * faction.experience[trait]
            )
            before = faction.traits[trait]
            delta = rate * (faction.experience[trait] - before)
            after = max(low, min(high, stream.noisy(before + delta, sigma)))
            faction.traits[trait] = after
            if abs(after - before) > 1e-6:
                moved[trait] = round(after - before, 4)

        if moved:
            events.append({"kind": "drift", "faction": faction.id, "moved": moved})
    return events
