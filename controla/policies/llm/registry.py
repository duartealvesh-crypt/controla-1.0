"""Registre des modèles : nom → `LLMBackend`.

Les identifiants ne vivent jamais dans le code ni dans le YAML : ils sont lus dans
l'environnement sous `{PROVIDER}_BASE_URL` / `{PROVIDER}_API_KEY`. Ajouter un
fournisseur, c'est ajouter deux variables d'environnement.

Un backend est **mis en cache par nom de modèle** et réutilisé d'une partie à
l'autre : un tournoi de 100 parties n'ouvre pas 100 connexions, et les statistiques
d'appel s'accumulent sur toute la campagne.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from controla.policies.llm.backend import LLMBackend

MODELS_PATH = Path(__file__).resolve().parent / "models.yaml"


def load_models(path: Path | str = MODELS_PATH) -> dict[str, dict]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def find_dotenv(start: Path | None = None) -> Path | None:
    """Cherche un `.env` en remontant depuis ce module.

    Une remontée plutôt qu'un `parents[N]` codé en dur : compter les niveaux se
    casse dès qu'un module bouge, et l'échec est alors silencieux — le modèle
    joue muet et on croit qu'il est prudent.
    """
    here = (start or Path(__file__).resolve()).parent
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv_if_present(start: Path | None = None) -> Path | None:
    """Charge le `.env` trouvé, s'il y en a un et si python-dotenv est installé."""
    try:
        from dotenv import load_dotenv
    except ImportError:                               # pragma: no cover
        return None
    found = find_dotenv(start)
    if found is not None:
        load_dotenv(found)
    return found


def provider_credentials(provider: str) -> tuple[str, str]:
    """`(base_url, api_key)` d'un fournisseur, ou une erreur explicite.

    Le message nomme les variables manquantes : une campagne qui échoue doit dire
    pourquoi en une ligne, pas laisser un modèle jouer muet.
    """
    prefix = provider.upper()
    names = (f"{prefix}_BASE_URL", f"{prefix}_API_KEY")
    values = tuple(os.environ.get(name) for name in names)
    missing = [name for name, value in zip(names, values) if not value]
    if missing:
        raise ValueError(
            f"identifiants manquants pour le fournisseur '{provider}' : "
            f"{', '.join(missing)}. Voir .env.example."
        )
    return values                                     # type: ignore[return-value]


def get_backend(
    model_name: str, models: dict[str, dict], cache: dict[str, LLMBackend], cfg=None
) -> LLMBackend:
    """Backend d'un modèle, créé à la première demande puis réutilisé."""
    if model_name in cache:
        return cache[model_name]
    if model_name not in models:
        raise KeyError(
            f"modèle inconnu : {model_name!r} — déclarés : {sorted(models)}"
        )

    entry = models[model_name]
    base_url, api_key = provider_credentials(entry["provider"])
    defaults = cfg.llm if cfg is not None else None

    def setting(key: str, fallback):
        if key in entry:
            return entry[key]
        return getattr(defaults, key) if defaults is not None else fallback

    backend = LLMBackend(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        tool_support=bool(entry.get("tool_support", True)),
        timeout=float(setting("timeout", 60.0)),
        temperature=setting("temperature", 0.7),
        max_tokens=int(setting("max_tokens", 2048)),
        seed=setting("seed", None),
        max_retries=int(setting("max_retries", 2)),
    )
    cache[model_name] = backend
    return backend
