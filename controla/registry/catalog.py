"""Le catalogue : toutes les instances de descripteurs du jeu.

`build_catalog(cfg)` est la SEULE fabrique. Le catalogue est une fonction de la
config, donc chaque coefficient affiché dans le prompt est celui que le moteur
applique réellement — impossible qu'un libellé dérive de la mécanique.

Ajouter du contenu (ressource, terrain, trait, bâtiment, palier, action) = ajouter
une entrée ici (+ ses constantes dans `config/default.yaml`). Il apparaît alors
automatiquement dans le prompt, dans `describe()` et dans le masque d'actions.
"""

from __future__ import annotations

from dataclasses import dataclass

from controla.registry.descriptors import (
    EFFECT_KINDS,
    ActionKind,
    ActionSpec,
    BuildingSpec,
    EffectSpec,
    ResearchTierSpec,
    ResourceSpec,
    TargetKind,
    TerrainSpec,
    TraitSpec,
)

# Clés de stat admises comme `driver` d'un bâtiment, en plus des clés de trait.
STAT_DRIVER_KEYS = frozenset({"population", "tiles", "building_density"})

# Ordre canonique des traits. Fige l'indexation du vecteur de traits dans
# l'observation — donc à ne PAS réordonner sans invalider les replays.
TRAIT_ORDER = ("builder", "warrior", "fertility", "resilient", "intelligence", "worker")


@dataclass(frozen=True)
class Catalog:
    resources: tuple[ResourceSpec, ...]
    terrains: tuple[TerrainSpec, ...]
    traits: tuple[TraitSpec, ...]
    buildings: tuple[BuildingSpec, ...]
    research_tiers: tuple[ResearchTierSpec, ...]
    actions: tuple[ActionSpec, ...]

    # ── index ────────────────────────────────────────────────────────────────

    def action(self, key: str) -> ActionSpec:
        for spec in self.actions:
            if spec.key == key:
                return spec
        raise KeyError(f"action inconnue : {key!r}")

    def building(self, key: str) -> BuildingSpec:
        for spec in self.buildings:
            if spec.key == key:
                return spec
        raise KeyError(f"bâtiment inconnu : {key!r}")

    @property
    def trait_keys(self) -> tuple[str, ...]:
        return tuple(spec.key for spec in self.traits)

    @property
    def terrain_keys(self) -> tuple[str, ...]:
        return tuple(spec.key for spec in self.terrains)

    @property
    def building_keys(self) -> tuple[str, ...]:
        return tuple(spec.key for spec in self.buildings)

    def trait_index(self, key: str) -> int:
        return self.trait_keys.index(key)

    def label(self, key: str) -> str:
        """Libellé lisible d'une clé, quelle que soit sa famille.

        Point d'entrée UNIQUE des libellés : les descripteurs s'en servent pour se
        décrire, plutôt que d'entretenir leurs propres tables — c'est ce qui évite
        que « Constructeur » vive à deux endroits et finisse par diverger.

        Les clés de stat (`population`, `tiles`, …) n'ont pas de descripteur ; on
        renvoie la clé, qui est déjà lisible.
        """
        for group in (self.traits, self.resources, self.terrains, self.buildings):
            for spec in group:
                if spec.key == key:
                    return spec.label
        for action in self.actions:
            if action.key == key:
                return action.label
        return key

    # ── validation ───────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Cohérence interne du catalogue. Appelé à chaque `build_catalog`, donc
        une incohérence de mod fait échouer le démarrage et non une partie."""
        trait_keys = set(self.trait_keys)
        resource_keys = {spec.key for spec in self.resources}

        for action in self.actions:
            for effect in action.effects:
                if effect.kind not in EFFECT_KINDS:
                    raise ValueError(
                        f"action '{action.key}' : effet de kind inconnu "
                        f"{effect.kind!r} (connus : {sorted(EFFECT_KINDS)})"
                    )
                if effect.kind in ("trait_call", "trait_call_foe"):
                    if effect.trait not in trait_keys:
                        raise ValueError(
                            f"action '{action.key}' : trait inconnu {effect.trait!r}"
                        )
            if action.kind is not ActionKind.CONTROL and action.cost < 1:
                raise ValueError(
                    f"action '{action.key}' : coût {action.cost} < 1 — toute action "
                    "doit avoir un coût d'opportunité réel (§6)"
                )

        for building in self.buildings:
            for driver in building.drivers:
                if driver not in trait_keys and driver not in STAT_DRIVER_KEYS:
                    raise ValueError(
                        f"bâtiment '{building.key}' : driver inconnu {driver!r} "
                        f"(traits : {sorted(trait_keys)} ; "
                        f"stats : {sorted(STAT_DRIVER_KEYS)})"
                    )
            if building.deficit_resource is not None:
                if building.deficit_resource not in resource_keys:
                    raise ValueError(
                        f"bâtiment '{building.key}' : deficit_resource inconnue "
                        f"{building.deficit_resource!r}"
                    )

        for terrain in self.terrains:
            unknown = set(terrain.produces) - resource_keys
            if unknown:
                raise ValueError(
                    f"terrain '{terrain.key}' : ressources inconnues {sorted(unknown)}"
                )

        levels = [tier.level for tier in self.research_tiers]
        if levels != sorted(levels) or len(set(levels)) != len(levels):
            raise ValueError(f"paliers de recherche non ordonnés/uniques : {levels}")


# ── Contenu ──────────────────────────────────────────────────────────────────

_RESOURCES = (
    ResourceSpec("food", "Nourriture", "nourrit le peuple chaque tour — c'est la survie"),
    ResourceSpec("wood", "Bois", "construction des bâtiments"),
    ResourceSpec("metal", "Métal", "équipe l'armée (bonus de Force) — débloqué par la Recherche"),
)

_TERRAIN_LABELS = {"plain": "Plaine", "river": "Rivière", "forest": "Forêt"}

_TRAITS = (
    TraitSpec("builder", "Constructeur", "envie de construire",
              ("constructions réalisées",)),
    TraitSpec("warrior", "Guerrier", "bonus de Force, envie de guerre, "
              "et fraction de population engagée au combat",
              ("guerres menées", "attaques subies")),
    TraitSpec("fertility", "Fertilité", "bonus/malus de natalité",
              ("croissance soutenue",)),
    TraitSpec("resilient", "Résilient", "bonus/malus de consommation de nourriture",
              ("famines répétées", "catastrophes subies")),
    TraitSpec("intelligence", "Intelligence", "bonus de recherche",
              ("laboratoires possédés",)),
    TraitSpec("worker", "Travailleur", "bonus de production sur toutes les ressources",
              ("richesse du terrain exploité", "bénédictions reçues")),
)

_BUILDING_LABELS = {
    "city": ("Ville", "rendement des cases ×{city_mult:.1f}, relève le plafond de population"),
    "lab": ("Laboratoire", "accélère la Recherche"),
    "camp": ("Camp militaire", "bonus de Force (saturant)"),
    "field": ("Champs", "rendement de nourriture ×{field_mult:.1f} sur la case"),
}

_RESEARCH_UNLOCKS = (
    "le Métal (production fixe proportionnelle au nombre de cases) — "
    "ouvre la route de la Force par la technologie",
    "le rendement agricole (+{tier2:.0%} de production de nourriture)",
    "la vision et une réduction de consommation (-{tier3:.0%})",
)


def _build_terrains(cfg) -> tuple[TerrainSpec, ...]:
    yields = cfg.production.terrain_yields
    return tuple(
        TerrainSpec(
            key=key,
            label=_TERRAIN_LABELS[key],
            produces=dict(yields[key]),
            buildable=True,
        )
        for key in _TERRAIN_LABELS
    )


def _build_buildings(cfg) -> tuple[BuildingSpec, ...]:
    specs = []
    for key, (label, effect_template) in _BUILDING_LABELS.items():
        block = cfg.buildings[key]
        specs.append(
            BuildingSpec(
                key=key,
                label=label,
                wood_cost=int(block["wood_cost"]),
                e_base=float(block["e_base"]),
                threshold=float(block["threshold"]),
                criticality_k=float(block["criticality_k"]),
                drivers=dict(block["drivers"]),
                deficit_resource=block["deficit_resource"],
                saturation_halflife=block["saturation_halflife"],
                effect=effect_template.format(
                    city_mult=cfg.production.city_multiplier,
                    field_mult=cfg.production.field_food_bonus,
                ),
            )
        )
    return tuple(specs)


def _build_research_tiers(cfg) -> tuple[ResearchTierSpec, ...]:
    thresholds = cfg.research.tiers
    return tuple(
        ResearchTierSpec(
            level=index + 1,
            threshold=float(threshold),
            unlocks=_RESEARCH_UNLOCKS[index].format(
                tier2=cfg.research.tier2_food_bonus,
                tier3=cfg.research.tier3_consumption_cut,
            ),
        )
        for index, threshold in enumerate(thresholds)
    )


def _build_actions(cfg) -> tuple[ActionSpec, ...]:
    costs = cfg.intervention.action_costs

    def cost(key: str) -> int:
        return int(costs[key])

    return (
        # ── Influences : indirectes, sur les traits de son propre peuple ──────
        ActionSpec(
            key="call_to_arms", label="Appel aux armes",
            kind=ActionKind.INFLUENCE, target_kind=TargetKind.NONE,
            cost=cost("call_to_arms"),
            effects=(
                EffectSpec("trait_call", trait="warrior", amount=2.0),
                EffectSpec("trait_call", trait="resilient", amount=-1.0),
            ),
        ),
        ActionSpec(
            key="call_to_moderation", label="Appel à la modération",
            kind=ActionKind.INFLUENCE, target_kind=TargetKind.NONE,
            cost=cost("call_to_moderation"),
            effects=(
                EffectSpec("trait_call", trait="resilient", amount=1.0),
                EffectSpec("trait_call", trait="warrior", amount=-2.0),
            ),
        ),
        ActionSpec(
            key="call_to_reflection", label="Appel à la réflexion",
            kind=ActionKind.INFLUENCE, target_kind=TargetKind.NONE,
            cost=cost("call_to_reflection"),
            effects=(
                EffectSpec("trait_call", trait="intelligence", amount=2.0),
                EffectSpec("trait_call", trait="worker", amount=1.0),
            ),
        ),
        ActionSpec(
            key="call_to_war", label="Appel à la guerre",
            kind=ActionKind.INFLUENCE, target_kind=TargetKind.FACTION,
            cost=cost("call_to_war"),
            effects=(EffectSpec("war_desire", amount=3.0),),
            note="l'envie monte proportionnellement à ton trait Guerrier ; "
                 "la guerre reste déclenchée par ton peuple, pas décrétée par toi",
        ),
        ActionSpec(
            key="call_to_trade", label="Appel au commerce",
            kind=ActionKind.INFLUENCE, target_kind=TargetKind.FACTION,
            cost=cost("call_to_trade"),
            effects=(EffectSpec("trade_relation", amount=cfg.trade.call_boost),),
            note=f"au-delà de {cfg.trade.threshold:.2f} les échanges deviennent "
                 "automatiques et profitables aux deux camps",
        ),
        ActionSpec(
            key="trade_embargo", label="Blocage au commerce",
            kind=ActionKind.INFLUENCE, target_kind=TargetKind.FACTION,
            cost=cost("trade_embargo"),
            effects=(EffectSpec("trade_relation", amount=-cfg.trade.embargo_drop),),
            note="arme économique : redoutable contre un rival dépendant du "
                 "commerce, inutile contre un autarcique",
        ),
        # ── Miracles : directs, sur le monde ─────────────────────────────────
        ActionSpec(
            key="blessing", label="Bénédiction des ressources",
            kind=ActionKind.MIRACLE, target_kind=TargetKind.OWN_TILE,
            cost=cost("blessing"),
            effects=(
                EffectSpec("tile_modifier", modifier="blessing",
                           amount=cfg.miracles.blessing_bonus,
                           duration_key="miracles.blessing_duration"),
                EffectSpec("trait_call", trait="worker", amount=1.0),
            ),
        ),
        ActionSpec(
            key="disaster", label="Désastre",
            kind=ActionKind.MIRACLE, target_kind=TargetKind.ENEMY_TILE,
            cost=cost("disaster"),
            effects=(
                EffectSpec("tile_modifier", modifier="disaster",
                           amount=-cfg.miracles.disaster_malus,
                           duration_key="miracles.disaster_duration"),
                EffectSpec("trait_call_foe", trait="resilient", amount=0.2),
            ),
        ),
        ActionSpec(
            key="omniscience", label="Omniscience",
            kind=ActionKind.MIRACLE, target_kind=TargetKind.NONE,
            cost=cost("omniscience"),
            effects=(EffectSpec("reveal", duration_key="miracles.omniscience_duration"),),
        ),
        # ── Diplomatie ───────────────────────────────────────────────────────
        ActionSpec(
            key="negotiate", label="Négocier",
            kind=ActionKind.DIPLOMACY, target_kind=TargetKind.FACTION_COMMITMENT,
            cost=cost("negotiate"),
            effects=(EffectSpec("open_channel"),),
            latency=int(cfg.diplomacy.negotiation_response_delay),
            carries_argument=True,
            note="joins un argumentaire libre et, si tu veux t'engager, un "
                 "commitment structuré (vérifiable par le moteur)",
        ),
        # ── Contrôle ─────────────────────────────────────────────────────────
        ActionSpec(
            key="end_turn", label="Fin de tour",
            kind=ActionKind.CONTROL, target_kind=TargetKind.NONE,
            cost=0,
            note="rends la main ; tes points d'intervention non dépensés sont perdus",
        ),
    )


def build_catalog(cfg) -> Catalog:
    """Construit et valide le catalogue pour une config donnée."""
    catalog = Catalog(
        resources=_RESOURCES,
        terrains=_build_terrains(cfg),
        traits=_TRAITS,
        buildings=_build_buildings(cfg),
        research_tiers=_build_research_tiers(cfg),
        actions=_build_actions(cfg),
    )
    catalog.validate()
    if catalog.trait_keys != TRAIT_ORDER:
        raise ValueError(
            f"l'ordre des traits doit rester {TRAIT_ORDER} (indexation de "
            f"l'observation), trouvé {catalog.trait_keys}"
        )
    return catalog


# ── Description des formules ─────────────────────────────────────────────────

def describe_formulas(cfg) -> dict[str, str]:
    """Les formules ne s'auto-décrivent pas champ par champ (§11) : elles sont
    décrites qualitativement, leurs coefficients étant lus dans la config.

    C'est la seule prose « en dur » autorisée du projet, et elle interpole
    systématiquement les constantes réelles plutôt que de les recopier.
    """
    return {
        "force": (
            "Force = Population × (base + Guerrier + apport métal + apport camps) "
            "× (1 − faim). Les apports technologiques sont SATURANTS : le métal "
            f"plafonne à +{cfg.force.metal_weight:.0%} (moitié atteinte à "
            f"{cfg.force.metal_halflife:.0f} métal), les camps à "
            f"+{cfg.force.camp_weight:.0%}. La faim est un malus multiplicatif, "
            "donc la Force ne peut jamais devenir négative. Deux routes vers la "
            "puissance : rush Guerrier (rapide) ou tech Métal (lente)."
        ),
        "engaged_force": (
            "Au combat, ton peuple engage une fraction de sa Force proportionnelle "
            f"à son trait Guerrier, entre {cfg.force.engaged_min:.0%} (peuple "
            f"modéré) et {cfg.force.engaged_max:.0%} (peuple très guerrier). Tu "
            "influences l'ardeur, jamais le nombre."
        ),
        "combat": (
            "P(l'attaquant gagne) suit un contest de Tullock : seul le RATIO des "
            f"forces compte, élevé à la puissance {cfg.combat.decisiveness:.1f}. "
            "Une armée deux fois supérieure gagne donc largement mais pas "
            "toujours. Une attaque NON DÉTECTÉE bénéficie d'un bonus de surprise "
            f"de ×{cfg.combat.surprise:.2f} — c'est la récompense d'une traîtrise "
            f"réussie. Une attaque met {cfg.combat.march_turns} tour(s) à arriver : "
            "c'est la fenêtre de réaction du défenseur, s'il la voit venir."
        ),
        "attrition": (
            "Le vainqueur saigne aussi. Les pertes dépendent du ratio de forces : "
            "une victoire écrasante coûte peu, une victoire de justesse coûte "
            "cher. C'est le frein anti-snowball — gagner toutes ses guerres en y "
            "perdant son peuple ne permet pas de rouler sur la carte."
        ),
        "population": (
            "La population suit une croissance logistique : rapide loin du "
            "plafond, elle se stabilise au plafond et décline au-dessus. Le "
            "plafond dépend de ta production de nourriture et de tes villes "
            f"(+{cfg.population.carrying_per_city:.0f} par ville). Autrement dit, "
            "la population monte jusqu'à ce que la nourriture la limite."
        ),
        "desire": (
            "Aucun bâtiment n'est ordonné : chacun émerge quand une envie franchit "
            "un seuil bruité. L'envie combine un plancher de construction, le "
            "déficit de la ressource concernée (lissé dans le temps, pour réagir "
            "à une tendance et non à un accident) et les traits pertinents. Entre "
            "envies concurrentes, l'arbitrage est stochastique et biaisé vers la "
            "plus forte."
        ),
        "drift": (
            "Tes traits DÉRIVENT vers ce que ton peuple vit réellement : guerres "
            "menées → Guerrier ↑, famines répétées → Résilient ↑, terrain riche "
            "→ Travailleur ↑. Tu ne choisis pas ce que devient ton peuple ; tes "
            "décisions en sont la pression de sélection."
        ),
        "intervention": (
            "Tes points d'intervention croissent avec ton peuple (population, "
            f"territoire, villes) mais plafonnent à {cfg.intervention.cap} : le "
            "pouvoir divin suit la foi sans s'emballer."
        ),
        "score": (
            "Score final = population × {population:.2f} + cases × {tiles:.2f} + "
            "recherche × {research:.2f} + bâtiments × {buildings:.2f}. Multi-axes "
            "volontairement : les voies économique, militaire et technologique "
            "restent toutes viables."
        ).format(**dict(cfg.score.weights)),
    }
