"""Chargement des constantes de jeu depuis un YAML, en objet immuable.

Un seul point d'entrée : `load_config()`. Le résultat est gelé — une constante
qui bouge en cours de partie casserait le déterminisme, donc toute écriture lève.

L'accès par attribut (`cfg.combat.decisiveness`) échoue fort et tôt sur une clé
inconnue, avec les clés voisines dans le message : une faute de frappe dans un
fichier de sweep ne doit jamais se dégrader silencieusement en valeur par défaut.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default.yaml"


class Config(Mapping):
    """Vue immuable, à accès par attribut, sur un dict de constantes imbriqué."""

    __slots__ = ("_data", "_path")

    def __init__(self, data: dict[str, Any], path: str = ""):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_path", path)

    # ── accès ────────────────────────────────────────────────────────────────

    def __getattr__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError:
            raise AttributeError(self._unknown_key_message(key)) from None
        if isinstance(value, dict):
            return Config(value, f"{self._path}{key}.")
        return value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def _unknown_key_message(self, key: str) -> str:
        known = ", ".join(sorted(self._data))
        return f"constante inconnue '{self._path}{key}' — clés disponibles ici : {known}"

    # ── immuabilité ──────────────────────────────────────────────────────────

    def __setattr__(self, key: str, value: Any) -> None:
        raise TypeError(
            f"Config est immuable (tentative d'écriture sur '{self._path}{key}'). "
            "Pour un sweep, charger un autre YAML via load_config(path=...) ou "
            "load_config(overrides=...)."
        )

    def __delattr__(self, key: str) -> None:
        raise TypeError("Config est immuable")

    # ── divers ───────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Copie profonde mutable — pour la sérialisation dans un log de partie."""
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({self._path or 'root'}: {', '.join(sorted(self._data))})"


def _deep_merge(base: dict, overrides: Mapping, path: str = "") -> dict:
    """Fusion récursive de `overrides` dans une copie de `base`.

    Une clé absente de `base` lève : un sweep ne doit pas pouvoir introduire une
    constante que le code ne lit jamais (typo silencieuse).
    """
    out = dict(base)
    for key, value in overrides.items():
        if key not in out:
            known = ", ".join(sorted(out))
            raise KeyError(
                f"override sur une constante inconnue '{path}{key}' — "
                f"clés disponibles ici : {known}"
            )
        if isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value, f"{path}{key}.")
        else:
            out[key] = value
    return out


def explode_dotted(flat: Mapping[str, Any]) -> dict:
    """`{"combat.attrition_loser": 0.2}` → `{"combat": {"attrition_loser": 0.2}}`.

    Les fichiers de sweep désignent une constante par son chemin complet — c'est
    la seule écriture lisible dans un tableau de variantes. La validation, elle,
    reste celle de `_deep_merge` : un chemin inconnu lève.
    """
    nested: dict = {}
    for dotted, value in flat.items():
        cursor = nested
        *parents, leaf = str(dotted).split(".")
        for part in parents:
            cursor = cursor.setdefault(part, {})
        cursor[leaf] = value
    return nested


def load_config(
    path: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Charge les constantes et renvoie une `Config` gelée.

    `path` défaut = `config/default.yaml`. `overrides` est un dict imbriqué
    fusionné par-dessus (c'est le mécanisme du sweep de la tranche 5).
    """
    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(source, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data)
