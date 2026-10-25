"""Banc de sélection de modèles : rapide, bon marché, et sur le VRAI prompt.

Pourquoi un banc avant les campagnes
------------------------------------
Choisir un modèle sur sa fiche ne suffit pas. Le catalogue d'OpenRouter annonce
`tool_choice` pour des modèles qui refusent en pratique un `tool_choice` NOMMÉ —
celui que `backend._call_with_tool` impose. Un tel modèle ne joue pas un seul
coup, si rapide soit-il. Et les modèles gratuits apparaissent, changent et
disparaissent d'un mois sur l'autre : la sélection doit pouvoir se REFAIRE.

Le banc répond à deux questions distinctes :

1. **Le code tourne-t-il avec ce modèle ?** — le tool call arrive-t-il, et sous
   une forme que la policy sait lire ;
2. **Le modèle comprend-il les règles ?** — ses index sont-ils dans la liste
   légale, et son plan tient-il dans ses points d'intervention.

Discipline de mesure
--------------------
* **Mêmes prompts pour tous.** Les sondes sont tirées de parties scriptées à
  seed fixée, à plusieurs moments de la partie (début, milieu, fin) : chaque
  modèle lit exactement les mêmes octets.
* **Premier essai seulement.** La policy relance une sortie invalide avec
  l'erreur précise ; le banc, non. Il mesure la conformité spontanée, qui est
  ce qui coûte des appels en campagne.
* **Quota d'abord.** Le palier gratuit plafonne à 20 requêtes/minute et 50 par
  jour sans crédit acheté. Le banc annonce son nombre d'appels AVANT de partir
  et se cadence sur la limite par minute.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from statistics import median

from controla.policies.llm.backend import CallStats, LLMBackend
from controla.policies.llm.prompt import PLAN_TOOL, build_system, build_user

CATALOGUE_URL = "https://openrouter.ai/api/v1/models"
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"

# Limites du palier gratuit d'OpenRouter (docs : api-reference/limits).
FREE_REQUESTS_PER_MINUTE = 20
FREE_REQUESTS_PER_DAY = 50              # 1 000 après 10 $ de crédits achetés

# Les méta-modèles d'OpenRouter (`openrouter/free`, `openrouter/auto`…) routent
# vers un modèle différent à chaque appel.
ROUTER_PREFIX = "openrouter/"


# ── Découverte ──────────────────────────────────────────────────────────────

def _get_json(url: str, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "controla"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _is_free(model: dict) -> bool:
    if model.get("id", "").endswith(":free"):
        return True
    pricing = model.get("pricing") or {}
    try:
        return float(pricing.get("prompt", 1)) == 0 and float(pricing.get("completion", 1)) == 0
    except (TypeError, ValueError):
        return False


def discover_free_candidates(min_uptime: float = 90.0, fetch=_get_json) -> list[dict]:
    """Les modèles gratuits jouables, d'après les données PUBLIQUES d'OpenRouter.

    Aucune clé nécessaire. Trois filtres, du moins cher au plus cher à vérifier :
    gratuit, `tools` + `tool_choice` annoncés, et disponibilité sur 24 h. Un
    modèle à 23 % de disponibilité perd trois parties sur quatre : inutile de le
    mesurer, il sortira de toute façon du banc.

    Ce filtre est une PRÉSÉLECTION. Le banc reste juge : la fiche dit ce que le
    fournisseur annonce, pas ce que le modèle fait.
    """
    candidates = []
    for model in fetch(CATALOGUE_URL).get("data", []):
        params = set(model.get("supported_parameters") or [])
        if not _is_free(model) or "tools" not in params or "tool_choice" not in params:
            continue
        if model["id"].startswith(ROUTER_PREFIX):
            # Un routeur choisit un modèle différent à chaque appel : le mesurer
            # ne dit rien de reproductible, et une partie y changerait d'adversaire
            # en cours de route.
            continue
        try:
            endpoints = fetch(ENDPOINTS_URL.format(model=model["id"]))["data"]["endpoints"]
        except Exception:                             # noqa: BLE001 — fiche incomplète
            endpoints = []
        uptimes = [e.get("uptime_last_1d") for e in endpoints if e.get("uptime_last_1d") is not None]
        uptime = max(uptimes) if uptimes else None
        if uptime is not None and uptime < min_uptime:
            continue
        candidates.append({
            "id": model["id"],
            "context_length": model.get("context_length"),
            "uptime_1d": uptime,
            "max_completion_tokens": max(
                (e.get("max_completion_tokens") or 0 for e in endpoints), default=None,
            ) or None,
            "reasoning": "reasoning" in params,
            "description": (model.get("description") or "")[:120],
        })
    return sorted(candidates, key=lambda entry: entry["id"])


# ── Sondes : les prompts réels sur lesquels tout le monde est mesuré ────────

@dataclass(frozen=True)
class Probe:
    """Un tour de jeu réel, figé : ce que le modèle lit, et de quoi le juger."""

    seed: int
    turn: int
    system: str
    user: str
    costs: dict[int, int]                 # index légal → coût en points
    points: int                           # points d'intervention disponibles

    @property
    def label(self) -> str:
        return f"seed {self.seed} · tour {self.turn}"


def build_probes(cfg, catalog, seeds=(1,), turns=(0, 6, 12),
                 seats=("cooperator", "conqueror", "expansionist", "turtle")) -> list[Probe]:
    """Fige la vue du siège 0 aux tours demandés, sur des parties scriptées.

    Scriptées pour être reproductibles : rejouer le banc dans un mois donne les
    mêmes prompts, donc des mesures comparables d'un mois sur l'autre.
    """
    from controla.env.controla_env import ControlaEnv, faction_id_of
    from controla.policies.archetypes import build_policy

    system = build_system(cfg, catalog)
    wanted = set(turns)
    probes: list[Probe] = []
    for seed in seeds:
        env = ControlaEnv(config=cfg)
        env.reset(seed=seed)
        policies = {agent: build_policy(name) for agent, name in zip(env.possible_agents, seats)}
        for agent, policy in policies.items():
            policy.reset(faction_id_of(agent), seed)
        taken: set[int] = set()
        for agent in env.agent_iter(max_iter=100_000):
            _obs, _reward, terminated, truncated, info = env.last()
            if terminated or truncated:
                env.step(None)
                continue
            view = info["view"]
            if agent == env.possible_agents[0] and view["turn"] in wanted - taken:
                taken.add(view["turn"])
                probes.append(Probe(
                    seed=seed, turn=view["turn"], system=system,
                    user=build_user(cfg, catalog, view),
                    costs={item["index"]: int(item["cost"]) for item in view["legal_actions"]},
                    points=int(view["self"]["intervention_points"]),
                ))
            info = {**info, "catalog": env.catalog, "index": env.index, "config": env.cfg}
            env.step(policies[agent].act(env.observe(agent), info))
            if taken == wanted:
                break
    return probes


# ── Jugement d'une réponse ──────────────────────────────────────────────────

def judge(payload: dict | None, probe: Probe) -> dict:
    """Ce qu'une réponse dit du modèle, critère par critère.

    Les critères sont emboîtés : un plan illisible n'a pas d'index à juger, un
    index illégal ne coûte rien. Chacun est donc rapporté séparément, pour
    distinguer « ne sait pas répondre » de « répond mais triche sur les règles ».
    """
    verdict = {"answered": payload is not None, "well_formed": False,
               "legal_rate": None, "within_budget": None, "moves": 0,
               "empty_plan": False, "has_reasoning": False}
    if payload is None:
        return verdict
    verdict["has_reasoning"] = bool(str(payload.get("reasoning") or "").strip())
    raw = payload.get("plan")
    if not isinstance(raw, list):
        return verdict
    indices = []
    for item in raw:
        if isinstance(item, int):
            indices.append(item)
        elif isinstance(item, dict) and isinstance(item.get("index"), int):
            indices.append(item["index"])
        else:
            return verdict                            # forme cassée : on s'arrête là
    verdict["well_formed"] = True
    verdict["moves"] = len(indices)
    verdict["empty_plan"] = not indices
    if indices:
        legal = [index for index in indices if index in probe.costs]
        verdict["legal_rate"] = len(legal) / len(indices)
        verdict["within_budget"] = sum(probe.costs[index] for index in legal) <= probe.points
    return verdict


# ── Le banc ─────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    model: str
    calls: list[dict] = field(default_factory=list)
    stats: CallStats = field(default_factory=CallStats)

    def summary(self) -> dict:
        calls = self.calls
        n = len(calls) or 1
        latencies = sorted(c["latency"] for c in calls if c["answered"])
        legal = [c["legal_rate"] for c in calls if c["legal_rate"] is not None]
        budget = [c["within_budget"] for c in calls if c["within_budget"] is not None]
        errors: dict[str, int] = {}
        for call in calls:
            if call["error"]:
                errors[call["error"]] = errors.get(call["error"], 0) + 1
        return {
            "model": self.model,
            "probes": len(calls),
            "answered": sum(c["answered"] for c in calls) / n,
            "well_formed": sum(c["well_formed"] for c in calls) / n,
            "legal_rate": sum(legal) / len(legal) if legal else None,
            "within_budget": sum(budget) / len(budget) if budget else None,
            "empty_plans": sum(c["empty_plan"] for c in calls),
            "latency_median": median(latencies) if latencies else None,
            "latency_max": latencies[-1] if latencies else None,
            "completion_tokens": self.stats.completion_tokens,
            "reasoning_tokens": self.stats.reasoning_tokens,
            "prompt_tokens": self.stats.prompt_tokens,
            "errors": errors,
            "finish_reasons": dict(self.stats.finish_reasons),
        }


def plan_size(models: list[str], probes: list[Probe]) -> int:
    return len(models) * len(probes)


def run_bench(backends: dict[str, LLMBackend], probes: list[Probe],
              requests_per_minute: float = FREE_REQUESTS_PER_MINUTE,
              on_call=None, sleep=time.sleep, clock=time.perf_counter) -> list[dict]:
    """Envoie chaque sonde à chaque modèle, une fois, au rythme permis.

    Séquentiel exprès : le quota gratuit est par minute ET par compte, donc
    paralléliser ne ferait que déclencher des 429 — que le SDK relancerait en
    gonflant les latences mesurées.
    """
    interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
    results = {name: ModelResult(model=name) for name in backends}
    last = None
    for probe in probes:
        for name, backend in backends.items():
            if last is not None and interval:
                wait = interval - (clock() - last)
                if wait > 0:
                    sleep(wait)
            last = clock()
            started = clock()
            payload = backend.complete(
                system=probe.system, user=probe.user, tool=PLAN_TOOL,
                sink=results[name].stats,
            )
            outcome = backend.last_call
            call = {
                "probe": probe.label,
                "latency": round(clock() - started, 3),
                "finish_reason": outcome.finish_reason,
                "error": outcome.error_kind,
                "native_reasoning_chars": len(outcome.native_reasoning or ""),
                **judge(payload, probe),
            }
            results[name].calls.append(call)
            if on_call is not None:
                on_call(name, probe, call)
    return [result.summary() | {"calls": result.calls} for result in results.values()]


def rank(summaries: list[dict]) -> list[dict]:
    """Ordre de recommandation : d'abord JOUER, ensuite respecter les règles,
    enfin aller vite. Un modèle rapide qui ne rend pas de plan ne vaut rien ;
    entre deux modèles conformes, le plus rapide l'emporte."""
    def key(summary: dict):
        return (
            -round(summary["well_formed"], 2),
            -round(summary["legal_rate"] or 0, 2),
            -round(summary["within_budget"] or 0, 2),
            summary["latency_median"] if summary["latency_median"] is not None else 1e9,
        )
    return sorted(summaries, key=key)


def make_backend(model: str, cfg, entry: dict | None = None) -> LLMBackend:
    """Un backend OpenRouter pour un identifiant quelconque — déclaré ou non.

    Le banc doit pouvoir mesurer un modèle AVANT qu'il n'entre dans
    `models.yaml` : c'est précisément pour décider s'il y entre.
    """
    from controla.policies.llm.registry import (
        api_model,
        provider_credentials,
        provider_headers,
    )

    entry = entry or {"provider": "openrouter"}
    provider = entry.get("provider", "openrouter")
    base_url, api_key = provider_credentials(provider)
    llm = cfg.llm

    def setting(key):
        return entry[key] if key in entry else getattr(llm, key)

    max_tokens = setting("max_tokens")
    return LLMBackend(
        base_url=base_url, api_key=api_key,
        model=api_model(model, entry), label=model,
        tool_support=bool(entry.get("tool_support", True)),
        timeout=float(setting("timeout")),
        temperature=setting("temperature"),
        max_tokens=None if max_tokens is None else int(max_tokens),
        seed=setting("seed"),
        # Pas de relance SDK : sur un 429 du palier gratuit, relancer fausserait
        # la latence mesurée et consommerait le quota du jour.
        max_retries=0,
        default_headers=provider_headers(provider),
        cache_control=bool(entry.get("cache_control", False)),
        reasoning_effort=entry.get("reasoning_effort"),
    )
