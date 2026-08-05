"""Rendu du règlement complet **depuis les seuls descripteurs**.

C'est le garde-fou de la discipline « moddable » : ce module ne connaît aucun nom
d'action, aucun coût, aucun libellé de bâtiment. Il itère le catalogue et appelle
`describe()`. Si une action n'apparaît pas ici, c'est qu'elle n'est pas déclarée —
et elle n'apparaîtra pas non plus dans le masque d'actions légales.

Le prompt LLM (tranche 2) réutilise exactement ces sections : la partie **statique**
du prompt est ce texte, stable sur toute une partie, donc cachable côté provider.
"""

from __future__ import annotations

from controla.registry.catalog import Catalog, describe_formulas
from controla.registry.descriptors import ActionKind

# L'ordre de la passe de développement (spec §8). L'ordre des opérations change
# l'équilibre du tout au tout, donc il est FIGÉ ici, épinglé par un test, et
# communiqué au dieu dans son prompt : il doit pouvoir raisonner dessus.
DEVELOPMENT_PASS_STEPS: tuple[tuple[str, str], ...] = (
    ("resolve_effects", "Résolution des effets échus (chantiers, armées arrivées, tributs)"),
    ("production", "Production par case : terrain × Travailleur × modificateurs, + aléa"),
    ("consumption", "Consommation de nourriture, modulée par Résilient"),
    ("population", "Croissance logistique de la population, plafonnée par la nourriture"),
    ("research", "Progression de la Recherche et franchissement des paliers"),
    ("force", "Recalcul de la Force militaire"),
    ("intervention", "Recharge des points d'intervention"),
    ("random_events", "Aléa : maladie, catastrophe naturelle, découverte"),
    ("emergence", "Comportements émergents : expansion, constructions, commerce, guerre"),
    ("war", "Résolution des batailles échues"),
    ("drift", "Dérive des traits sous la pression du vécu"),
    ("decay", "Expiration des modificateurs temporaires"),
    ("elimination", "Éliminations (population épuisée ou territoire perdu)"),
    ("observe", "Reward et nouvelles observations"),
)

_ACTION_SECTIONS: tuple[tuple[ActionKind, str, str], ...] = (
    (
        ActionKind.INFLUENCE,
        "INFLUENCES — indirectes, sur les traits de ton peuple",
        "Tu ne donnes aucun ordre : tu inclines des traits, et ton peuple agit de "
        "lui-même dans la passe de développement.",
    ),
    (
        ActionKind.MIRACLE,
        "MIRACLES — directs, sur le monde",
        "Seuls tes miracles agissent directement sur les cases.",
    ),
    (
        ActionKind.DIPLOMACY,
        "DIPLOMATIE",
        "Tu peux mener plusieurs négociations en parallèle avec différents dieux.",
    ),
    (
        ActionKind.CONTROL,
        "CONTRÔLE",
        "",
    ),
)


def _section(title: str, lines: list[str], preamble: str = "") -> str:
    head = f"═══ {title} " + "═" * max(0, 62 - len(title))
    body = [head]
    if preamble:
        body.append(preamble)
    body.extend(lines)
    return "\n".join(body)


def render_rules(cfg, catalog: Catalog) -> str:
    """Le règlement complet, dérivé du catalogue et de la config."""
    formulas = describe_formulas(cfg)
    blocks: list[str] = []

    blocks.append(
        f"CONTROLA — arène divine à {cfg.world.num_factions} dieux.\n"
        f"Grille {cfg.world.grid_size}×{cfg.world.grid_size}, "
        f"{cfg.world.max_turns} tours, départs strictement symétriques.\n"
        "Tu règnes sur un peuple semi-autonome. Tu agis DIRECTEMENT sur le monde, "
        "INDIRECTEMENT sur ton peuple."
    )

    blocks.append(_section(
        "RESSOURCES",
        [f"  • {spec.describe(cfg, catalog)}" for spec in catalog.resources],
    ))

    blocks.append(_section(
        "TERRAINS",
        [f"  • {spec.describe(cfg, catalog)}" for spec in catalog.terrains],
        "Spécialisation stricte. Le métal ne vient d'aucun terrain.",
    ))

    blocks.append(_section(
        "TRAITS DE TON PEUPLE",
        [f"  • {spec.describe(cfg, catalog)}" for spec in catalog.traits],
        "Un trait joue trois rôles : statistique lue par la passe, moteur qui "
        "déclenche les comportements, et valeur qui dérive sous pression.",
    ))

    blocks.append(_section(
        "BÂTIMENTS",
        [f"  • {spec.describe(cfg, catalog)}" for spec in catalog.buildings],
        formulas["desire"],
    ))

    blocks.append(_section(
        "RECHERCHE",
        [f"  • {spec.describe(cfg, catalog)}" for spec in catalog.research_tiers],
        formulas["research"],
    ))

    for kind, title, preamble in _ACTION_SECTIONS:
        specs = [spec for spec in catalog.actions if spec.kind is kind]
        if not specs:
            continue
        blocks.append(_section(
            title, [f"  • {spec.describe(cfg, catalog)}" for spec in specs], preamble
        ))

    blocks.append(_section(
        "POINTS D'INTERVENTION",
        [f"  {formulas['intervention']}"],
    ))

    blocks.append(_section(
        "FORCE ET COMBAT",
        [
            f"  {formulas['force']}",
            f"  {formulas['engaged_force']}",
            f"  {formulas['combat']}",
            f"  {formulas['attrition']}",
            f"  {formulas['conquest']}",
        ],
    ))

    blocks.append(_section(
        "POPULATION ET DÉRIVE",
        [
            f"  {formulas['population']}",
            f"  {formulas['expansion']}",
            f"  {formulas['drift']}",
        ],
    ))

    blocks.append(_section(
        "ORDRE DE LA PASSE DE DÉVELOPPEMENT",
        [
            f"  {index:2d}. {label}"
            for index, (_key, label) in enumerate(DEVELOPMENT_PASS_STEPS, start=1)
        ],
        "Cet ordre est figé. Il détermine ce qui se répercute sur quoi — "
        "raisonne dessus.",
    ))

    blocks.append(_section(
        "SCORE ET FIN DE PARTIE",
        [
            f"  {formulas['score']}",
            f"  Fin au tour {cfg.world.max_turns}, ou dès qu'il ne reste qu'un "
            "survivant.",
            "  Élimination : population épuisée OU plus aucun territoire.",
            "  Les 4 dieux sont classés — il n'y a pas de match nul.",
        ],
    ))

    blocks.append(_section(
        "RÉPUTATION",
        [
            "  Tu peux toujours ne pas honorer ce que tu as promis. Le coût est "
            "social : un compteur public de promesses tenues et trahies, visible "
            "par TOUS les dieux et injecté dans leurs observations.",
            "  Une promesse est jugée sur l'action que tu as PAYÉE, pas sur son "
            "issue : tenir parole et perdre la bataille reste tenir parole.",
        ],
    ))

    return "\n\n".join(blocks)
