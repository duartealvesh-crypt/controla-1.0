"""Descripteurs déclaratifs : la couche « moddable par construction » (spec §11).

Discipline non négociable : **les effets sont de la donnée introspectable**, jamais
des lambdas opaques. `EffectSpec(kind="trait_call", trait="warrior", amount=2.0)`
peut être *exécuté* par `actions/apply.py` ET *auto-décrit* par `policies/llm/prompt.py`.
Un effet caché derrière une closure serait exécutable mais indescriptible, et
« moddable » deviendrait un vœu.

Ajouter un mod = ajouter un descripteur. Il apparaît alors automatiquement dans le
prompt et dans le masque d'actions légales, sans toucher au moteur.

Les *formules* (combat, croissance logistique, envies) ne s'auto-décrivent pas
champ par champ : elles sont décrites qualitativement, leurs coefficients étant
lus dans la config. Voir `describe_formulas()` dans catalog.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# Les kinds d'effet reconnus. C'est le contrat entre le descripteur (qui décrit) et
# `actions/apply.py` (qui exécute) : ce module assure que tout kind se décrit, et
# apply.py assure à l'import que tout kind a un interpréteur. Ajouter un kind sans
# interpréteur fait donc échouer le démarrage, jamais une partie en cours.
EFFECT_KINDS = frozenset({
    "trait_call",
    "trait_call_foe",
    "war_desire",
    "trade_relation",
    "tile_modifier",
    "reveal",
    "open_channel",
})


class TargetKind(str, Enum):
    """Ce que l'action prend pour cible. Détermine le domaine d'index statique
    de l'espace d'action plat (voir actions/legality.py)."""

    NONE = "none"            # action globale, sans cible
    OWN_TILE = "own_tile"    # une case possédée par l'acteur
    ENEMY_TILE = "enemy_tile"  # une case possédée par quelqu'un d'autre
    FACTION = "faction"      # une autre faction
    # Une autre faction ET le type d'engagement joint au message. Replier
    # l'engagement dans l'index le rend MASQUABLE : on ne peut pas promettre
    # une alliance de guerre à une faction éliminée.
    FACTION_COMMITMENT = "faction_commitment"


# Les engagements qu'un dieu peut joindre à une négociation (§7). Vérifiables
# par le moteur, donc la trahison est détectable sans ambiguïté.
NO_COMMITMENT = "none"
WAR_ALLIANCE = "war_alliance"
ECON_ALLIANCE = "econ_alliance"
COMMITMENT_KINDS: tuple[str, ...] = (NO_COMMITMENT, WAR_ALLIANCE, ECON_ALLIANCE)


class ActionKind(str, Enum):
    """Famille d'action — sert au regroupement dans le prompt, pas à la mécanique."""

    INFLUENCE = "influence"    # indirect, sur les traits de son peuple
    MIRACLE = "miracle"        # direct, sur le monde
    DIPLOMACY = "diplomacy"    # canal de négociation
    CONTROL = "control"        # end_turn : rend la main, ne coûte rien


# ── Effets ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EffectSpec:
    """Un effet élémentaire, en données pures.

    `kind` sélectionne l'interpréteur dans `actions/apply.py` ; les autres champs
    sont sa charge utile. Un `kind` inconnu doit faire échouer le moteur au
    démarrage (cf. `Catalog.validate`), jamais silencieusement à l'exécution.

    Champs par kind :
      trait_call     trait, amount        — déplace un trait de l'acteur (saturant)
      trait_call_foe trait, amount        — déplace un trait de la cible
      war_desire     amount               — pousse l'envie de guerre vers la cible
      trade_relation amount               — déplace la relation commerciale (signé)
      tile_modifier  modifier, duration_key — pose un modificateur daté sur la case
      reveal         duration_key         — lève le brouillard N tours
      open_channel   —                    — ouvre/poursuit une négociation
    """

    kind: str
    trait: str | None = None
    amount: float = 0.0
    modifier: str | None = None
    duration_key: str | None = None

    def describe(self, cfg, catalog) -> str:
        """Phrase lisible par un humain — et par un LLM. Les coefficients sont
        tirés de la config et les libellés du catalogue, jamais réécrits ici."""
        if self.kind in ("trait_call", "trait_call_foe"):
            who = "ton peuple" if self.kind == "trait_call" else "le peuple ciblé"
            arrow = _arrow(self.amount)
            return f"{catalog.label(self.trait)} {arrow} sur {who}"
        if self.kind == "war_desire":
            return f"envie de guerre envers la cible {_arrow(self.amount)}"
        if self.kind == "trade_relation":
            verb = "monte" if self.amount > 0 else "chute"
            return f"la relation commerciale avec la cible {verb} ({self.amount:+.2f})"
        if self.kind == "tile_modifier":
            duration = _resolve(cfg, self.duration_key)
            sign = "hausse" if self.amount > 0 else "baisse"
            return (
                f"productivité de la case en {sign} de {abs(self.amount):.0%} "
                f"pendant {duration:.0f} tours"
            )
        if self.kind == "reveal":
            duration = _resolve(cfg, self.duration_key)
            return f"brouillard levé pendant {duration:.0f} tour(s) (carte + intérieurs)"
        if self.kind == "open_channel":
            return "ouvre un canal de négociation (réponse au tour suivant)"
        raise ValueError(f"EffectSpec.kind inconnu : {self.kind!r}")


def _arrow(amount: float) -> str:
    """Flèches proportionnelles — le dieu doit pouvoir prédire l'effet de son
    influence (§0.3 : on cache les coefficients, jamais les causes).

    Les paliers sont calés sur les intensités de la spec §6 : un Appel aux armes
    (2.0) donne « ↑↑ », un Appel à la guerre (3.0) « ↑↑↑ ».
    """
    magnitude = abs(amount)
    if magnitude >= 3.0:
        arrows = "↑↑↑"
    elif magnitude >= 2.0:
        arrows = "↑↑"
    elif magnitude >= 0.5:
        arrows = "↑"
    else:
        arrows = "↑ (très légèrement)"
    return arrows if amount > 0 else arrows.replace("↑", "↓")


def _resolve(cfg, dotted_key: str | None) -> float:
    """Lit `a.b.c` dans la config. Les descripteurs référencent les constantes par
    leur chemin, pour qu'aucune valeur numérique ne soit dupliquée ici."""
    if dotted_key is None:
        raise ValueError("duration_key manquant")
    node = cfg
    for part in dotted_key.split("."):
        node = getattr(node, part)
    return float(node)


# ── Ressources, terrains, traits ─────────────────────────────────────────────

@dataclass(frozen=True)
class ResourceSpec:
    key: str
    label: str
    role: str

    def describe(self, cfg, catalog) -> str:
        return f"{self.label} — {self.role}"


@dataclass(frozen=True)
class TerrainSpec:
    key: str
    label: str
    produces: dict[str, float]
    buildable: bool = True

    def describe(self, cfg, catalog) -> str:
        yields = ", ".join(
            f"{amount:.1f} {key}" for key, amount in self.produces.items() if amount > 0
        ) or "rien"
        suffix = "" if self.buildable else " (non constructible)"
        return f"{self.label} — produit {yields}/tour/case{suffix}"


@dataclass(frozen=True)
class TraitSpec:
    key: str
    label: str
    effect: str
    pressure_sources: tuple[str, ...] = ()

    def describe(self, cfg, catalog) -> str:
        bounds = f"[{cfg.traits.min:+.0%}, {cfg.traits.max:+.0%}]"
        pressure = ", ".join(self.pressure_sources) or "—"
        return f"{self.label} — {self.effect} · bornes {bounds} · monte sous : {pressure}"



# ── Bâtiments & recherche ────────────────────────────────────────────────────

@dataclass(frozen=True)
class BuildingSpec:
    """Un bâtiment n'est jamais « ordonné » : il émerge quand une envie franchit
    un seuil bruité. Tous les paramètres de cette envie sont ici, en données."""

    key: str
    label: str
    wood_cost: int
    e_base: float
    threshold: float
    criticality_k: float
    drivers: dict[str, float]
    deficit_resource: str | None = None
    saturation_halflife: float | None = None
    effect: str = ""

    def describe(self, cfg, catalog) -> str:
        drivers = ", ".join(
            f"{catalog.label(name)} ↑" for name in self.drivers
        )
        deficit = (
            f" · s'intensifie sous déficit de {self.deficit_resource}"
            if self.deficit_resource else ""
        )
        effect = f" · {self.effect}" if self.effect else ""
        return (
            f"{self.label} — {self.wood_cost} bois · envie portée par {drivers}"
            f"{deficit}{effect}"
        )


@dataclass(frozen=True)
class ResearchTierSpec:
    level: int
    threshold: float
    unlocks: str

    def describe(self, cfg, catalog) -> str:
        return f"Palier {self.level} ({self.threshold:.0f} pts) — {self.unlocks}"


# ── Actions du dieu ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActionSpec:
    """Une action du dieu. `effects` est la liste complète et lisible de ce qu'elle
    fait ; `latency` (générique, pas de cas spécial) diffère la résolution."""

    key: str
    label: str
    kind: ActionKind
    target_kind: TargetKind
    cost: int
    effects: tuple[EffectSpec, ...] = ()
    latency: int = 0
    carries_argument: bool = False
    note: str = ""

    def describe(self, cfg, catalog) -> str:
        target = _TARGET_LABELS[self.target_kind]
        effects = " · ".join(
            effect.describe(cfg, catalog) for effect in self.effects
        ) or "—"
        latency = f" · effet dans {self.latency} tour(s)" if self.latency else ""
        note = f" · {self.note}" if self.note else ""
        return (
            f"{self.key} ({self.cost} pt{'s' if self.cost > 1 else ''}, {target}) "
            f"→ {effects}{latency}{note}"
        )


_TARGET_LABELS = {
    TargetKind.NONE: "sans cible",
    TargetKind.OWN_TILE: "une de tes cases",
    TargetKind.ENEMY_TILE: "une case ennemie",
    TargetKind.FACTION: "une autre faction",
    TargetKind.FACTION_COMMITMENT: "une autre faction, avec ou sans engagement",
}
