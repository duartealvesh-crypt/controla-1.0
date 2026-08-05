"""Le panel scripté : ce que chaque archétype déclare doit être ce qu'il joue.

Un archétype exprime ses préférences sur des **types d'effet**, jamais sur des noms
d'action. Le risque propre à ce design est qu'un effet change de sens selon son
signe et que l'archétype ne puisse pas exprimer la différence — c'est exactement ce
qui est arrivé aux miracles, et ces tests l'interdisent désormais.
"""

from __future__ import annotations

import pytest

from controla.arena.storage import game_files, load_match
from controla.policies.archetypes import ARCHETYPES, ScriptedPolicy
from controla.registry.catalog import build_catalog
from controla.registry.descriptors import ActionKind
from tests.helpers import base_config

PACIFISTS = ("cooperator", "turtle", "conqueror", "expansionist")
HOSTILE_KEYS = ("war_desire", "embargo", "curse")


@pytest.fixture(scope="module")
def catalog():
    return build_catalog(base_config())


def keys_of(spec) -> set[str]:
    return {ScriptedPolicy._effect_key(effect) for effect in spec.effects}


def test_every_signed_effect_has_its_own_weighting_key(catalog):
    """Un effet dont le signe inverse le SENS doit produire deux clés distinctes.

    Sinon l'archétype pondère « enrichir sa case » et « ruiner celle du voisin » du
    même poids, et sa conduite déclarée ne veut plus rien dire. `trait_call` fait
    exception : son signe est traité dans `_score` — un trait gagné attire, le même
    trait perdu décourage — et le test suivant l'exige.
    """
    seen: dict[str, set[bool]] = {}
    for spec in catalog.actions:
        for effect in spec.effects:
            if effect.amount is None:
                continue
            seen.setdefault(ScriptedPolicy._effect_key(effect), set()).add(
                effect.amount > 0
            )

    ambiguous = {key for key, signs in seen.items() if len(signs) > 1}
    assert ambiguous <= {"trait_call"}, (
        f"ces clés recouvrent un effet et son contraire : {sorted(ambiguous)} — "
        "ajoute une clé signée dans ScriptedPolicy._effect_key"
    )


def test_losing_a_valued_trait_discourages_the_action(catalog):
    """Un Appel monte un trait ET en baisse un autre : +2 Guerrier pour −1
    Résilience. Ne lire que le versant positif rendait l'archétype aveugle à ce
    qu'il sacrifie."""
    view = {"rivals": [], "commitments": [], "inbox": [], "turn": 1}

    def score(name: str, action_key: str) -> float:
        spec = next(s for s in catalog.actions if s.key == action_key)
        return ScriptedPolicy(name)._score({"spec": spec, "target": None}, view)

    # Le technologue tient à sa Résilience (1.2) : l'Appel aux armes la lui coûte.
    assert score("turtle", "call_to_arms") < score("turtle", "call_to_reflection")
    # Le conquérant tient à son trait Guerrier (1.5) : la modération le lui coûte.
    assert score("conqueror", "call_to_moderation") < score("conqueror", "call_to_arms")


@pytest.mark.parametrize("name", PACIFISTS)
def test_a_peaceful_archetype_prefers_the_blessing_to_the_curse(name, catalog):
    """Le coopérateur lançait 219 malédictions sur 48 parties — 12 % de son budget
    divin — alors qu'il déclare `war_desire −0.9` et `embargo −0.9`."""
    policy = ScriptedPolicy(name)
    view = {"rivals": [], "commitments": [], "inbox": [], "turn": 1}

    def score(action_key: str) -> float:
        spec = next(s for s in catalog.actions if s.key == action_key)
        return policy._score({"spec": spec, "target": (0, 0)}, view)

    assert score("blessing") > score("disaster"), (
        f"{name} pondère la malédiction comme la bénédiction"
    )


def test_an_aggressive_archetype_still_reaches_for_the_curse(catalog):
    """La correction ne doit pas éteindre l'action pour tout le monde : sans un
    archétype qui maudit, le miracle négatif ne serait plus jamais exercé."""
    policy = ScriptedPolicy("betrayer")
    view = {"rivals": [], "commitments": [], "inbox": [], "turn": 1}
    spec = next(s for s in catalog.actions if s.key == "disaster")
    neutral = next(s for s in catalog.actions if s.key == "omniscience")
    assert (
        policy._score({"spec": spec, "target": (0, 0)}, view)
        > policy._score({"spec": neutral, "target": None}, view)
    )


def test_no_archetype_weights_an_action_by_name():
    """La discipline qui rend le panel moddable : un mod qui ajoute une action
    belliqueuse doit être adopté sans recâblage."""
    catalog_keys = {spec.key for spec in build_catalog(base_config()).actions}
    for name, profile in ARCHETYPES.items():
        for table in (profile.kind_weights, profile.effect_weights,
                      profile.trait_weights):
            leaked = catalog_keys & set(table)
            assert not leaked, f"{name} pondère des noms d'action : {sorted(leaked)}"


def test_the_action_families_used_as_weights_all_exist():
    families = {kind.value for kind in ActionKind}
    for name, profile in ARCHETYPES.items():
        unknown = set(profile.kind_weights) - families
        assert not unknown, f"{name} pondère des familles inconnues : {sorted(unknown)}"
