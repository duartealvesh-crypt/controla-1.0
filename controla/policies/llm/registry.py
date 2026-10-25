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


def provider_headers(provider: str) -> dict[str, str]:
    """En-têtes optionnels d'un fournisseur, lus eux aussi dans l'environnement.

    Seul OpenRouter en demande aujourd'hui (attribution : `HTTP-Referer` et
    `X-Title`). Ils sont purement décoratifs — un en-tête absent ne change ni la
    réponse ni la facturation — d'où l'absence de toute erreur si les variables
    ne sont pas renseignées.
    """
    if provider.lower() != "openrouter":
        return {}
    headers = {}
    for variable, header in (
        ("OPENROUTER_APP_URL", "HTTP-Referer"),
        ("OPENROUTER_APP_NAME", "X-Title"),
    ):
        value = os.environ.get(variable)
        if value:
            headers[header] = value
    return headers


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


# Les clés qu'une entrée de `models.yaml` peut porter. Toute autre clé est une
# ERREUR, pas un avertissement : `memroy: true` serait sinon ignoré en silence,
# et l'étude mémoire comparerait deux variantes… sans mémoire.
ENTRY_KEYS = frozenset({
    "provider", "tool_support", "model", "memory", "cache_control",
    "temperature", "max_tokens", "timeout", "seed", "max_retries",
})


def api_model(seat: str, entry: dict) -> str:
    """L'identifiant envoyé à l'API. Par défaut, le nom du siège lui-même.

    Le champ `model` permet de déclarer plusieurs SIÈGES pour un même modèle —
    `openai/gpt-5-nano` et `openai/gpt-5-nano@memoire` — qui seront classés
    séparément tout en appelant la même chose.
    """
    return str(entry.get("model") or seat)


def validate_entry(seat: str, entry: dict) -> None:
    unknown = set(entry) - ENTRY_KEYS
    if unknown:
        raise ValueError(
            f"models.yaml, '{seat}' : clé(s) inconnue(s) {sorted(unknown)} — "
            f"admises : {sorted(ENTRY_KEYS)}"
        )
    if "provider" not in entry:
        raise ValueError(f"models.yaml, '{seat}' : `provider` manquant")
    for flag in ("memory", "cache_control", "tool_support"):
        if flag in entry and not isinstance(entry[flag], bool):
            raise ValueError(
                f"models.yaml, '{seat}' : `{flag}` doit valoir true ou false, "
                f"reçu {entry[flag]!r}"
            )


def memory_enabled(entry: dict, cfg=None) -> bool:
    """Mémoire inter-tours d'un siège : l'entrée prime, sinon le défaut global."""
    if "memory" in entry:
        return bool(entry["memory"])
    try:
        return bool(cfg.llm.memory) if cfg is not None else False
    except AttributeError:
        return False


def get_backend(
    model_name: str, models: dict[str, dict], cache: dict[str, LLMBackend], cfg=None
) -> LLMBackend:
    """Backend d'un siège, créé à la première demande puis réutilisé.

    Mis en cache par SIÈGE et non par identifiant API : deux variantes d'un même
    modèle ont chacune leurs statistiques d'appel, sans quoi on ne pourrait pas
    comparer ce que la mémoire coûte en tokens.
    """
    if model_name in cache:
        return cache[model_name]
    if model_name not in models:
        raise KeyError(
            f"modèle inconnu : {model_name!r} — déclarés : {sorted(models)}"
        )

    entry = models[model_name]
    validate_entry(model_name, entry)
    base_url, api_key = provider_credentials(entry["provider"])
    defaults = cfg.llm if cfg is not None else None

    def setting(key: str, fallback):
        if key in entry:
            return entry[key]
        return getattr(defaults, key) if defaults is not None else fallback

    max_tokens = setting("max_tokens", 2048)
    backend = LLMBackend(
        base_url=base_url,
        api_key=api_key,
        model=api_model(model_name, entry),
        label=model_name,
        cache_control=bool(entry.get("cache_control", False)),
        tool_support=bool(entry.get("tool_support", True)),
        timeout=float(setting("timeout", 60.0)),
        temperature=setting("temperature", 0.7),
        # `None` reste `None` : un modèle peut légitimement vouloir qu'aucun
        # plafond ne soit envoyé, et `int(None)` planterait la campagne.
        max_tokens=None if max_tokens is None else int(max_tokens),
        seed=setting("seed", None),
        max_retries=int(setting("max_retries", 2)),
        default_headers=provider_headers(entry["provider"]),
    )
    cache[model_name] = backend
    return backend
