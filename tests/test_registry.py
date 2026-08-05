"""Le catalogue s'auto-décrit intégralement, et rien ne le duplique.

Ces tests sont le garde-fou de la discipline « moddable par construction » (§11) :
un mod ajouté au catalogue doit apparaître tout seul dans le règlement, et aucune
table de libellés ou de coûts ne doit vivre en parallèle du catalogue.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from controla.config.loader import load_config
from controla.registry.catalog import (
    STAT_DRIVER_KEYS,
    TRAIT_ORDER,
    build_catalog,
    describe_formulas,
)
from controla.registry.descriptors import EFFECT_KINDS, ActionKind
from controla.rules_text import DEVELOPMENT_PASS_STEPS, render_rules

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "controla"


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def catalog(cfg):
    return build_catalog(cfg)


# ── auto-description ─────────────────────────────────────────────────────────

def test_every_descriptor_describes_itself(cfg, catalog):
    """Aucun descripteur ne peut exister sans savoir se décrire."""
    groups = (
        catalog.resources, catalog.terrains, catalog.traits,
        catalog.buildings, catalog.research_tiers, catalog.actions,
    )
    for group in groups:
        assert group, "groupe de descripteurs vide"
        for spec in group:
            text = spec.describe(cfg, catalog)
            assert text and not text.isspace(), f"{spec} ne se décrit pas"


def test_every_action_appears_in_the_rules(cfg, catalog):
    """Une action absente du règlement serait invisible au dieu."""
    rules = render_rules(cfg, catalog)
    for action in catalog.actions:
        assert action.key in rules, f"action '{action.key}' absente du règlement"


def test_every_building_and_trait_appears_in_the_rules(cfg, catalog):
    rules = render_rules(cfg, catalog)
    for spec in (*catalog.buildings, *catalog.traits, *catalog.terrains):
        assert spec.label in rules, f"'{spec.label}' absent du règlement"


def test_rules_carry_the_real_coefficients():
    """Les coefficients affichés sont ceux de la config, pas des copies figées."""
    tweaked = load_config(overrides={"intervention": {"cap": 42}})
    rules = render_rules(tweaked, build_catalog(tweaked))
    assert "plafonnent à 42" in rules, "le plafond n'est pas interpolé depuis la config"

    hot = load_config(overrides={"trade": {"embargo_drop": 0.99}})
    hot_catalog = build_catalog(hot)
    assert "-0.99" in hot_catalog.action("trade_embargo").describe(hot, hot_catalog)


def test_pass_order_is_pinned():
    """L'ordre de la passe change l'équilibre du tout au tout (§8) : on l'épingle.

    Ce test doit échouer si quelqu'un réordonne les étapes sans le décider — pas
    parce que l'ordre est sacré, mais parce que le changer est une décision
    d'équilibrage à assumer, jamais un effet de bord.
    """
    assert [key for key, _label in DEVELOPMENT_PASS_STEPS] == [
        "resolve_effects", "production", "consumption", "population",
        "research", "force", "intervention", "random_events",
        "emergence", "war", "drift", "decay", "elimination", "observe",
    ]


# ── cohérence interne ────────────────────────────────────────────────────────

def test_trait_order_is_stable(catalog):
    """L'indexation du vecteur de traits dans l'observation en dépend."""
    assert catalog.trait_keys == TRAIT_ORDER


def test_effect_kinds_are_all_declared(catalog):
    used = {
        effect.kind
        for action in catalog.actions
        for effect in action.effects
    }
    assert used <= EFFECT_KINDS
    unused = EFFECT_KINDS - used
    assert not unused, f"kinds déclarés mais jamais utilisés : {sorted(unused)}"


def test_action_costs_come_from_config(cfg, catalog):
    """Chaque action non-contrôle a un coût lu dans la config, et >= 1 (§6)."""
    declared = dict(cfg.intervention.action_costs)
    for action in catalog.actions:
        if action.kind is ActionKind.CONTROL:
            continue
        assert action.key in declared, f"coût manquant pour '{action.key}'"
        assert action.cost == declared[action.key]
        assert action.cost >= 1


def test_building_drivers_reference_known_keys(catalog):
    known = set(catalog.trait_keys) | STAT_DRIVER_KEYS
    for building in catalog.buildings:
        assert set(building.drivers) <= known


def test_metal_comes_from_no_terrain(catalog):
    """Spécialisation stricte : le métal ne vient que de la Recherche (§2)."""
    for terrain in catalog.terrains:
        assert terrain.produces.get("metal", 0) == 0


def test_first_research_tier_unlocks_metal(catalog):
    assert "Métal" in catalog.research_tiers[0].unlocks


def test_validate_rejects_an_unknown_effect_kind(cfg, catalog):
    from dataclasses import replace

    from controla.registry.descriptors import EffectSpec

    broken = replace(
        catalog,
        actions=catalog.actions[:-1] + (
            replace(catalog.action("omniscience"), effects=(EffectSpec("teleport"),)),
        ),
    )
    with pytest.raises(ValueError, match="kind inconnu"):
        broken.validate()


def test_validate_rejects_an_unknown_building_driver(cfg, catalog):
    from dataclasses import replace

    broken = replace(
        catalog,
        buildings=(replace(catalog.building("lab"), drivers={"charisma": 1.0}),),
    )
    with pytest.raises(ValueError, match="driver inconnu"):
        broken.validate()


def test_all_formulas_are_rendered(cfg, catalog):
    """Toute formule décrite doit finir dans le règlement — sinon elle est morte."""
    rules = render_rules(cfg, catalog)
    for key, text in describe_formulas(cfg).items():
        assert text in rules, f"formule '{key}' décrite mais absente du règlement"


# ── pas de duplication ───────────────────────────────────────────────────────

def test_no_hardcoded_action_names_outside_the_catalog():
    """Un nom d'action ne doit vivre qu'au catalogue (+ config + tests).

    C'est le mode de défaite qu'on refuse de reproduire : dans la v1,
    `prompt_builder.py` maintenait à la main des tables de coûts et de libellés
    que `config.py` connaissait déjà — elles dérivaient en silence.
    """
    cfg = load_config()
    catalog = build_catalog(cfg)
    action_keys = [action.key for action in catalog.actions]

    allowed = {"registry/catalog.py", "rules_text.py"}
    offenders: dict[str, list[str]] = {}
    for path in SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        if relative in allowed:
            continue
        source = path.read_text(encoding="utf-8")
        # On cherche le nom entre quotes : une utilisation en littéral de chaîne.
        found = [
            key for key in action_keys
            if re.search(rf"""["']{re.escape(key)}["']""", source)
        ]
        if found:
            offenders[relative] = found

    assert not offenders, (
        "noms d'action écrits en dur hors du catalogue : "
        + "; ".join(f"{path} -> {keys}" for path, keys in offenders.items())
    )


def test_labels_live_only_in_the_catalog(catalog):
    """Aucun libellé ne doit être recopié hors du catalogue.

    `Catalog.label()` est le point d'entrée unique : les descripteurs s'en servent
    pour se décrire au lieu d'entretenir leur propre table. Sans ce test, une
    seconde table de traits réapparaîtrait tôt ou tard et divergerait en silence.
    """
    labels = [
        spec.label
        for group in (catalog.traits, catalog.resources, catalog.buildings, catalog.terrains)
        for spec in group
    ]
    offenders: dict[str, list[str]] = {}
    for path in SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        if relative == "registry/catalog.py":
            continue
        source = path.read_text(encoding="utf-8")
        found = [label for label in labels if f'"{label}"' in source]
        if found:
            offenders[relative] = found

    assert not offenders, (
        "libellés recopiés hors du catalogue : "
        + "; ".join(f"{path} -> {found}" for path, found in offenders.items())
    )


def test_catalog_label_resolves_every_family(catalog):
    assert catalog.label("warrior") == "Guerrier"
    assert catalog.label("food") == "Nourriture"
    assert catalog.label("city") == "Ville"
    assert catalog.label("forest") == "Forêt"
    # Une clé de stat n'a pas de descripteur : elle est déjà lisible.
    assert catalog.label("population") == "population"


def test_no_magic_numbers_in_the_catalog():
    """Le catalogue est de la structure ; les nombres vivent dans le YAML.

    On tolère les intensités d'Appel (amount=2.0 : « ↑↑ » est une propriété de
    l'action, pas une constante d'équilibrage) et les petits entiers de structure.
    """
        # `amount=` porte des intensités assumées ; on inspecte le reste.
    source = (SOURCE_ROOT / "registry" / "catalog.py").read_text(encoding="utf-8")
    stripped = re.sub(r"amount=-?\d+\.?\d*", "", source)
    stripped = re.sub(r"#.*", "", stripped)
    stripped = re.sub(r'""".*?"""', "", stripped, flags=re.S)
    literals = re.findall(r"(?<![\w.])\d+\.\d+", stripped)
    assert not literals, f"constantes numériques en dur dans catalog.py : {literals}"
