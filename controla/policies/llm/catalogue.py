"""Audit du registre de modèles contre ce que le fournisseur expose réellement.

Pourquoi ce module existe : `models.yaml` est une **déclaration**, et une
déclaration se périme en silence. Un identifiant de modèle retiré du catalogue
produit une erreur `NotFound` classée « fatale » — au mieux au bout de quinze
tours joués pour rien, au pire sur un siège seulement, laissant croire à un
modèle prudent. Un réglage d'échantillonnage que le modèle n'accepte pas est
pire encore : il est ignoré sans bruit, et le classement compare alors des
conditions d'inférence différentes.

L'audit interroge donc `{base_url}/models`, endpoint que toutes les API
compatibles OpenAI exposent, et confronte trois choses :

1. **l'existence** du modèle dans le catalogue du fournisseur ;
2. **les paramètres acceptés**, quand le fournisseur les publie
   (`supported_parameters`, propre à OpenRouter mais absent ailleurs) ;
3. **le prix**, pour chiffrer une campagne avant de la lancer plutôt qu'après.

Le module ne corrige rien : il rapporte. Choisir de retirer un modèle ou de
mettre un réglage à `null` reste une décision, pas un effet de bord.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from controla.policies.llm.registry import (
    api_model,
    provider_credentials,
    provider_headers,
    validate_entry,
)

# Réglages d'inférence que `models.yaml` peut porter et qui sont susceptibles
# d'être refusés par un modèle donné. `max_tokens` et `timeout` n'y sont pas :
# le premier est universel, le second est côté client.
SAMPLING_KEYS = ("temperature", "seed", "top_p")

# Une partie à 4 sièges LLM coûte ~180 000 tokens (FINDINGS n° 5), répartis
# ~90 % en entrée et ~10 % en sortie sur les mesures de T2.
TOKENS_PER_SEAT_GAME = 45_000
INPUT_SHARE = 0.9


def fetch_provider_models(provider: str, timeout: float = 30.0) -> dict[str, dict]:
    """Catalogue d'un fournisseur, indexé par identifiant de modèle.

    Renvoie un dict vide si le fournisseur n'expose pas l'endpoint : tous les
    serveurs compatibles OpenAI ne l'implémentent pas, et son absence n'est pas
    une panne — elle veut seulement dire qu'on ne pourra rien vérifier.
    """
    base_url, api_key = provider_credentials(provider)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "controla",
            **provider_headers(provider),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return {
        entry["id"]: entry
        for entry in payload.get("data", [])
        if isinstance(entry, dict) and "id" in entry
    }


def _price(entry: dict, key: str) -> float | None:
    """Prix par million de tokens, ou `None` si le fournisseur ne le publie pas."""
    try:
        return float((entry.get("pricing") or {})[key]) * 1e6
    except (KeyError, TypeError, ValueError):
        return None


def audit_model(name: str, declared: dict, catalogue: dict[str, dict]) -> dict:
    """Confronte une entrée de `models.yaml` au catalogue du fournisseur.

    `name` est le SIÈGE ; l'identifiant cherché dans le catalogue est celui que
    l'API recevra (`model:` d'une variante, sinon le siège lui-même).
    """
    target = api_model(name, declared)
    entry = catalogue.get(target)
    report: dict = {
        "model": name,
        "api_model": target,
        "provider": declared.get("provider"),
        "memory": bool(declared.get("memory", False)),
        "cache_control": bool(declared.get("cache_control", False)),
        "known": entry is not None,
        "problems": [],
        "warnings": [],
    }
    try:
        validate_entry(name, declared)
    except ValueError as error:
        report["problems"].append(str(error))
    if entry is None:
        # Sans catalogue on ne peut pas conclure : `catalogue` vide veut dire
        # « endpoint muet », pas « modèle inexistant ». Mais l'absence de
        # problème n'est PAS un satisfecit — un audit qui n'a rien pu vérifier
        # doit le dire, sinon il rassure exactement quand il ne sait rien.
        if catalogue:
            report["problems"].append(
                "absent du catalogue du fournisseur — ce modèle ne jouera jamais"
            )
        else:
            report["warnings"].append(
                "catalogue du fournisseur injoignable — rien n'a pu être vérifié"
            )
        return report

    report["context_length"] = entry.get("context_length")
    report["price_in"] = _price(entry, "prompt")
    report["price_out"] = _price(entry, "completion")

    supported = entry.get("supported_parameters")
    if supported is None:
        report["warnings"].append(
            "le fournisseur ne publie pas la liste des paramètres acceptés — "
            "réglages d'inférence invérifiables"
        )
        return report
    supported = set(supported)

    if declared.get("tool_support", True) and "tools" not in supported:
        report["problems"].append(
            "`tool_support: true` déclaré, mais le modèle n'annonce pas `tools` — "
            "passer à false pour basculer sur le schéma JSON annexé"
        )

    for key in SAMPLING_KEYS:
        # Un réglage explicitement mis à `null` est une abstention assumée :
        # il n'y a rien à vérifier.
        if key not in declared or declared[key] is None:
            continue
        if key not in supported:
            report["problems"].append(
                f"`{key}` déclaré mais non supporté — il serait ignoré en silence, "
                f"et le panel comparerait des conditions d'inférence différentes "
                f"(mettre `{key}: null`)"
            )
    return report


def audit(models: dict[str, dict], providers: dict[str, dict[str, dict]]) -> list[dict]:
    """Audite tout le registre. `providers` : catalogue déjà chargé par fournisseur."""
    return [
        audit_model(name, declared or {}, providers.get(declared.get("provider", ""), {}))
        for name, declared in sorted(models.items())
    ]


def campaign_cost(report: dict, games: int, seats: int = 1) -> float | None:
    """Coût estimé d'une campagne pour ce modèle, en dollars.

    Une estimation grossière et **annoncée comme telle** : elle extrapole les
    45 000 tokens par siège et par partie mesurés en T2, sans tenir compte du
    cache ni de la verbosité propre au modèle. Elle sert à distinguer « quelques
    centimes » de « quelques centaines de dollars » avant de lancer, pas à
    facturer.
    """
    price_in, price_out = report.get("price_in"), report.get("price_out")
    if price_in is None or price_out is None:
        return None
    total = TOKENS_PER_SEAT_GAME * games * seats
    return (
        total * INPUT_SHARE * price_in + total * (1 - INPUT_SHARE) * price_out
    ) / 1e6


def load_provider_catalogues(models: dict[str, dict]) -> tuple[dict[str, dict], dict[str, str]]:
    """Charge le catalogue de chaque fournisseur cité. Renvoie `(catalogues, erreurs)`.

    Une panne sur un fournisseur n'empêche pas d'auditer les autres : c'est
    précisément quand une clé manque qu'on a besoin de voir le reste du rapport.
    """
    catalogues: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for provider in sorted({entry.get("provider", "") for entry in models.values() if entry}):
        if not provider:
            continue
        try:
            catalogues[provider] = fetch_provider_models(provider)
        except ValueError as error:                   # identifiants manquants
            errors[provider] = str(error)
        except urllib.error.HTTPError as error:
            errors[provider] = f"HTTP {error.code} sur /models — clé refusée ?"
        except Exception as error:                    # noqa: BLE001 — réseau, DNS, TLS…
            errors[provider] = f"{type(error).__name__}: {error}"
    return catalogues, errors
