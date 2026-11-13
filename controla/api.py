"""Le point d'entrée du moteur pour les outils EXTERNES (équilibrage, RL, analyse).

Un outil d'optimisation ne doit connaître du jeu que ceci : « joue une partie avec
ces sièges, cette seed et ces constantes, rends-moi le résultat ». Tout le reste —
environnement, policies, catalogue — est un détail d'implémentation qu'il n'a pas à
importer, pour que le moteur puisse changer sans casser l'outillage.

Le contrat :

* les entrées et les sorties sont des **types Python simples** (str, int, float,
  list, dict), sérialisables en JSON ;
* le moteur **n'importe jamais** l'outillage ; un test le vérifie.
"""

from __future__ import annotations

import dataclasses
import json

from controla.arena.play import MatchSpec, play_match
from controla.config.loader import load_config
from controla.policies.archetypes import (
    ARCHETYPES,
    SCRIPTED_NAMES,
    Archetype,
    ScriptedPolicy,
)
from controla.policies.factory import build_policies
from controla.registry.catalog import build_catalog

SEATS_PER_GAME = 4


def scripted_names() -> list[str]:
    """Les sièges scriptés disponibles (archétypes, `random_legal`, `noop`)."""
    return list(SCRIPTED_NAMES)


def archetype_params(name: str) -> dict:
    """Les paramètres d'un archétype, comme point de départ d'une optimisation.

    Le dict rendu peut être modifié puis repassé tel quel comme siège de
    `play_game`.
    """
    if name not in ARCHETYPES:
        raise KeyError(f"archétype inconnu : {name!r} — disponibles : {sorted(ARCHETYPES)}")
    return dataclasses.asdict(ARCHETYPES[name])


def play_game(
    seats: list[str | dict],
    seed: int,
    overrides: dict | None = None,
) -> dict:
    """Joue une partie complète et rend son résultat en données simples.

    `seats` : 4 sièges, chacun soit un **nom** (archétype scripté ou modèle déclaré
    dans `models.yaml`), soit un **dict de paramètres d'archétype** (même forme que
    `archetype_params`, avec une clé `name`).

    `overrides` : constantes à modifier, en dict imbriqué
    (`{"world": {"grid_size": 10}}`), comme pour un sweep.

    Le résultat contient le classement final et, pour chaque tour, les événements,
    les batailles et un instantané du plateau (propriétaires, scores, guerres).
    """
    if len(seats) != SEATS_PER_GAME:
        raise ValueError(f"il faut {SEATS_PER_GAME} sièges, reçu {len(seats)}")
    cfg = load_config(overrides=overrides)
    catalog = build_catalog(cfg)

    named = [seat for seat in seats if isinstance(seat, str)]
    built = iter(build_policies(named, cfg, catalog))
    policies = [
        next(built) if isinstance(seat, str) else ScriptedPolicy(Archetype(**seat))
        for seat in seats
    ]
    labels = [seat if isinstance(seat, str) else str(seat["name"]) for seat in seats]

    result = play_match(MatchSpec(seed=seed, seats=labels, snapshots=True), policies,
                        config=cfg)
    # Aller-retour JSON : tuples, clés entières… le résultat devient exactement ce
    # qu'un fichier rendrait, sans objet du moteur qui traîne.
    return json.loads(json.dumps(result))
