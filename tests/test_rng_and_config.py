"""L'aléa est keyé, reproductible entre processus, et insensible à l'ordre d'appel.

C'est l'exigence d'équité du benchmark (§11) : un dieu qui joue cinq actions ne doit
pas décaler l'aléa subi par ses adversaires. Un test vérifie aussi que le dérivé de
clé survit à un autre processus — c'est ce que `hash()` ne garantit PAS sur les `str`,
son hachage étant salé par processus.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from controla.config.loader import load_config
from controla.rng import KeyedRNG, derive_seed


# ── dérivé de clé ────────────────────────────────────────────────────────────

def test_derive_seed_is_pure():
    assert derive_seed(1, 2, "production", 3) == derive_seed(1, 2, "production", 3)


@pytest.mark.parametrize(
    "a, b",
    [
        ((1, 2, "production", 3), (1, 2, "production", 4)),   # faction
        ((1, 2, "production", 3), (1, 3, "production", 3)),   # tour
        ((1, 2, "production", 3), (1, 2, "war", 3)),          # purpose
        ((1, 2, "production", 3), (2, 2, "production", 3)),   # seed
        ((1, 2, "production", None), (1, 2, "production", 0)),  # None != faction 0
    ],
)
def test_derive_seed_separates_every_key_component(a, b):
    assert derive_seed(*a) != derive_seed(*b)


def test_derive_seed_is_stable_across_processes():
    """Le test qui justifie blake2b plutôt que hash()."""
    code = (
        "from controla.rng import derive_seed;"
        "print(derive_seed(42, 7, 'production', 1))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    )
    assert int(out.stdout.strip()) == derive_seed(42, 7, "production", 1)


# ── flux ─────────────────────────────────────────────────────────────────────

def test_the_same_key_returns_the_same_continuing_stream():
    """Redemander une clé POURSUIT la séquence, ne la redémarre pas — sinon deux
    appels dans la même passe tireraient la même valeur."""
    rng = KeyedRNG(1)
    first = rng.stream(1, "production", 0).random()
    second = rng.stream(1, "production", 0).random()
    assert first != second


def test_streams_are_independent_across_keys():
    """LE test d'équité : consommer de l'aléa pour une faction ne doit rien changer
    pour les autres."""
    quiet = KeyedRNG(9)
    expected = quiet.stream(3, "production", 1).random()

    busy = KeyedRNG(9)
    for _ in range(50):                       # la faction 0 s'agite
        busy.stream(3, "production", 0).random()
    busy.stream(3, "war", 0).gauss(0, 1)
    assert busy.stream(3, "production", 1).random() == expected


def test_two_rngs_with_the_same_seed_agree():
    a, b = KeyedRNG(4), KeyedRNG(4)
    assert [a.stream(t, "x").random() for t in range(5)] == \
           [b.stream(t, "x").random() for t in range(5)]


def test_different_seeds_diverge():
    a, b = KeyedRNG(4), KeyedRNG(5)
    assert a.stream(0, "x").random() != b.stream(0, "x").random()


# ── primitives ───────────────────────────────────────────────────────────────

def test_noisy_with_zero_sigma_consumes_no_randomness():
    """Une config déterministe ne doit pas décaler la séquence."""
    rng = KeyedRNG(2)
    stream = rng.stream(0, "x")
    assert stream.noisy(5.0, 0.0) == 5.0
    assert stream.random() == KeyedRNG(2).stream(0, "x").random()


def test_weighted_choice_respects_the_weights():
    stream = KeyedRNG(3).stream(0, "x")
    picks = [stream.weighted_choice(["a", "b"], [1.0, 0.0]) for _ in range(30)]
    assert set(picks) == {"a"}


def test_weighted_choice_falls_back_to_uniform_on_null_weights():
    stream = KeyedRNG(3).stream(0, "x")
    picks = {stream.weighted_choice(["a", "b", "c"], [0.0, 0.0, 0.0]) for _ in range(60)}
    assert picks == {"a", "b", "c"}


def test_softmax_choice_at_zero_temperature_is_argmax():
    stream = KeyedRNG(3).stream(0, "x")
    for _ in range(10):
        assert stream.softmax_choice(["a", "b", "c"], [0.1, 5.0, 0.2], 0.0) == "b"


def test_softmax_choice_stays_stochastic_at_high_temperature():
    stream = KeyedRNG(3).stream(0, "x")
    picks = {stream.softmax_choice(["a", "b"], [1.0, 1.2], 2.0) for _ in range(60)}
    assert picks == {"a", "b"}


def test_softmax_choice_survives_large_scores():
    """Le max est soustrait avant l'exponentielle : pas d'overflow."""
    stream = KeyedRNG(3).stream(0, "x")
    assert stream.softmax_choice(["a", "b"], [1e6, 1e6 + 5], 1.0) in {"a", "b"}


def test_softmax_choice_rejects_an_empty_population():
    stream = KeyedRNG(3).stream(0, "x")
    with pytest.raises(ValueError, match="population vide"):
        stream.softmax_choice([], [], 1.0)


# ── journal ──────────────────────────────────────────────────────────────────

def test_journal_is_off_by_default_and_records_when_enabled():
    quiet = KeyedRNG(1)
    quiet.stream(0, "x").random()
    assert quiet.journal == []

    loud = KeyedRNG(1, journal_enabled=True)
    loud.stream(0, "x").gauss(0, 1)
    assert len(loud.journal) == 1
    assert loud.journal[0]["call"] == "gauss"
    assert loud.journal[0]["key"] == "T0:x"


def test_journal_digest_detects_a_different_trajectory():
    def trace(seed, draws):
        rng = KeyedRNG(seed, journal_enabled=True)
        for index in range(draws):
            rng.stream(index, "x").random()
        return rng.journal_digest()

    assert trace(1, 5) == trace(1, 5)
    assert trace(1, 5) != trace(1, 6)
    assert trace(1, 5) != trace(2, 5)


# ── config ───────────────────────────────────────────────────────────────────

def test_config_is_immutable():
    cfg = load_config()
    with pytest.raises(TypeError, match="immuable"):
        cfg.world = None


def test_unknown_constant_fails_loudly_with_neighbours():
    cfg = load_config()
    with pytest.raises(AttributeError) as excinfo:
        _ = cfg.combat.decisivenes          # faute de frappe
    message = str(excinfo.value)
    assert "combat.decisivenes" in message
    assert "decisiveness" in message, "le message doit lister les clés voisines"


def test_overrides_merge_deeply_without_touching_siblings():
    cfg = load_config(overrides={"combat": {"decisiveness": 9.0}})
    assert cfg.combat.decisiveness == 9.0
    assert cfg.combat.surprise == load_config().combat.surprise


def test_override_on_an_unknown_key_is_rejected():
    """Une typo dans un fichier de sweep ne doit jamais devenir un silence."""
    with pytest.raises(KeyError, match="combat.decisivenes"):
        load_config(overrides={"combat": {"decisivenes": 9.0}})


def test_to_dict_is_a_deep_copy():
    cfg = load_config()
    dumped = cfg.to_dict()
    dumped["combat"]["decisiveness"] = 999
    assert cfg.combat.decisiveness != 999
