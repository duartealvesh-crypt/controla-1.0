"""Fabrique unique de policies : un nom de siège → une `Policy`.

Le vocabulaire est le même partout (CLI, tournoi, sweep) : soit le nom d'un
archétype scripté, soit une entrée de `policies/llm/models.yaml`. L'arène ignore
laquelle des deux familles occupe un siège — c'est l'étanchéité des couches (§11).
"""

from __future__ import annotations

from controla.policies.archetypes import SCRIPTED_NAMES, build_policy as build_scripted
from controla.policies.base import Policy


def build_policies(
    seats: list[str], cfg, catalog, backend_cache: dict | None = None
) -> list[Policy]:
    """Construit une policy par siège.

    `backend_cache` est partagé entre parties : un tournoi de 100 parties n'ouvre
    qu'un client par modèle, et les statistiques d'appel s'accumulent sur toute la
    campagne plutôt que d'être remises à zéro à chaque partie.
    """
    cache = backend_cache if backend_cache is not None else {}
    models: dict[str, dict] | None = None
    policies: list[Policy] = []

    for name in seats:
        if name in SCRIPTED_NAMES:
            policies.append(build_scripted(name))
            continue

        # Import différé : une campagne 100 % scriptée ne doit pas exiger `openai`.
        from controla.policies.llm.policy import LLMPolicy
        from controla.policies.llm.registry import (
            get_backend,
            load_dotenv_if_present,
            load_models,
        )

        if models is None:
            load_dotenv_if_present()
            models = load_models()
        policies.append(
            LLMPolicy(get_backend(name, models, cache, cfg), cfg, catalog)
        )
    return policies


def available_seats() -> dict[str, list[str]]:
    """Ce qu'on peut assigner à un siège, par famille."""
    try:
        from controla.policies.llm.registry import load_models
        models = sorted(load_models())
    except Exception:                                  # noqa: BLE001 — YAML absent/illisible
        models = []
    return {"scripted": list(SCRIPTED_NAMES), "models": models}
