"""Une stratégie = un vecteur réel x, que CMA-ES peut faire varier librement.

L'archétype du moteur porte des poids de natures différentes : positifs
(familles d'actions, appétence d'engagement), signés (effets, traits), taux dans
[0, 1]. On les ramène tous à des réels sans contrainte :

* poids positif  w = exp(x)       (x = 0 ↔ poids neutre 1)
* poids signé    w = x            (x = 0 ↔ indifférent)
* taux           r = sigmoïde(x)  (x = 0 ↔ 0,5)

`x` est borné à ±CLIP pour qu'un pas de CMA-ES trop grand ne produise pas un
poids absurde (exp(4) ≈ 55 suffit à rendre une action quasi exclusive).

La température d'échantillonnage et la règle de cible ne sont pas optimisées :
la première est fixée, la seconde est un choix discret fait par exécution.
"""

from __future__ import annotations

import math

from controla import api

CLIP = 4.0
TEMPERATURE = 0.35
RATE_EPS = 1e-3


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(rate: float) -> float:
    rate = min(1 - RATE_EPS, max(RATE_EPS, rate))
    return math.log(rate / (1 - rate))


class Space:
    """Correspondance entre vecteur x et dict d'archétype pour `api.play_game`.

    `frozen_diplomacy` : propose et accepte toujours à 0 — le jeu sans diplomatie,
    sans toucher au moteur. Les coordonnées diplomatiques sont alors retirées de x.
    """

    def __init__(self, frozen_diplomacy: bool = False, target_rule: str = "none"):
        space = api.archetype_space()
        if target_rule not in space["target_rules"]:
            raise ValueError(f"target_rule inconnue : {target_rule!r}")
        self.kinds = space["kinds"]
        self.effects = space["effects"]
        self.traits = space["traits"]
        self.commitment_kinds = space["commitment_kinds"]
        self.frozen = frozen_diplomacy
        self.target_rule = target_rule

        self.names = (
            [f"kind:{k}" for k in self.kinds]
            + [f"effect:{e}" for e in self.effects]
            + [f"trait:{t}" for t in self.traits]
        )
        if not self.frozen:
            self.names += [f"commit:{c}" for c in self.commitment_kinds]
            self.names += ["propose_rate", "accept_rate", "honour_rate"]

    @property
    def dim(self) -> int:
        return len(self.names)

    # ── archétype → x ───────────────────────────────────────────────────────

    def encode(self, params: dict) -> list[float]:
        commitment = params.get("commitment_rate", 0.0)
        values = {
            **{f"kind:{k}": math.log(max(1e-6, params["kind_weights"].get(k, 1.0)))
               for k in self.kinds},
            **{f"effect:{e}": params["effect_weights"].get(e, 0.0) for e in self.effects},
            **{f"trait:{t}": params["trait_weights"].get(t, 0.0) for t in self.traits},
            **{f"commit:{c}": math.log(max(1e-6, params["commitment_kinds"].get(c, 1.0)))
               for c in self.commitment_kinds},
            "propose_rate": _logit(_rate(params, "propose_rate", commitment)),
            "accept_rate": _logit(_rate(params, "accept_rate", commitment)),
            "honour_rate": _logit(params.get("honour_rate", 1.0)),
        }
        return [_clip(values[name]) for name in self.names]

    # ── x → archétype ───────────────────────────────────────────────────────

    def decode(self, x, name: str) -> dict:
        v = dict(zip(self.names, (_clip(float(value)) for value in x)))
        params = {
            "name": name,
            "kind_weights": {k: math.exp(v[f"kind:{k}"]) for k in self.kinds},
            "effect_weights": {e: v[f"effect:{e}"] for e in self.effects},
            "trait_weights": {t: v[f"trait:{t}"] for t in self.traits},
            "target_rule": self.target_rule,
            "temperature": TEMPERATURE,
            "commitment_rate": 0.0,
        }
        if self.frozen:
            params.update(propose_rate=0.0, accept_rate=0.0, honour_rate=1.0,
                          commitment_kinds={c: 1.0 for c in self.commitment_kinds})
        else:
            params.update(
                commitment_kinds={c: math.exp(v[f"commit:{c}"]) for c in self.commitment_kinds},
                propose_rate=_sigmoid(v["propose_rate"]),
                accept_rate=_sigmoid(v["accept_rate"]),
                honour_rate=_sigmoid(v["honour_rate"]),
            )
        return params

    def adapt(self, params: dict) -> dict:
        """Un archétype existant, rendu conforme à cet espace (diplomatie gelée
        si besoin). Ses poids d'origine sont conservés tels quels."""
        params = dict(params)
        if self.frozen:
            params.update(commitment_rate=0.0, propose_rate=0.0, accept_rate=0.0)
        return params


def _rate(params: dict, key: str, default: float) -> float:
    value = params.get(key)
    return default if value is None else value


def _clip(value: float) -> float:
    return max(-CLIP, min(CLIP, value))


def describe(params: dict, top: int = 6) -> str:
    """Les traits saillants d'une stratégie, en une ligne lisible."""
    parts = []
    kinds = sorted(params["kind_weights"].items(), key=lambda kv: -kv[1])
    parts.append("familles " + ", ".join(f"{k} {w:.2f}" for k, w in kinds))
    effects = sorted(params["effect_weights"].items(), key=lambda kv: -abs(kv[1]))[:top]
    parts.append("effets " + ", ".join(f"{k} {w:+.2f}" for k, w in effects))
    traits = sorted(params["trait_weights"].items(), key=lambda kv: -abs(kv[1]))[:3]
    parts.append("traits " + ", ".join(f"{k} {w:+.2f}" for k, w in traits))
    propose = params.get("propose_rate")
    if propose is not None:
        parts.append(
            f"propose {propose:.2f} · accepte {params.get('accept_rate', 0):.2f}"
            f" · parole {params.get('honour_rate', 1):.2f}"
        )
    return " | ".join(parts)
