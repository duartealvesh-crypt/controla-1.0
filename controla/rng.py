"""Aléa keyé et auditable (spec §11).

Deux exigences du benchmark, toutes deux non négociables :

1. **Flux indépendants keyés par `(seed, turn, purpose, faction)`, jamais par
   ordre d'appel.** Sinon deux modèles subissent un aléa différent selon leur
   nombre d'actions, et le classement mesure la chance. Un dieu qui joue 5 actions
   ne doit pas décaler la récolte de son voisin.

2. **Chaque tirage journalisable**, pour que les parties soient auditables et
   rejouables.

Le dérivé de clé est `blake2b`, **pas** `hash()` : le hachage des `str` est salé
par processus en Python, donc `hash()` produirait un non-déterminisme silencieux
entre deux exécutions — exactement le bug qu'un benchmark ne peut pas se permettre.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence


def derive_seed(seed: int, turn: int, purpose: str, faction: int | None = None) -> int:
    """Entier 64 bits stable dérivé de la clé. Reproductible entre processus,
    machines et versions de Python."""
    key = f"{seed}|{turn}|{purpose}|{'' if faction is None else faction}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


@dataclass
class Stream:
    """Un flux de tirages pour une clé donnée. Journalise chaque appel si demandé.

    N'expose que les primitives réellement utilisées par le moteur : ajouter une
    primitive ici est un choix conscient, ce qui garde la surface d'aléa lisible.
    """

    key: str
    _rng: random.Random
    _journal: list[dict] | None = None

    # ── primitives ───────────────────────────────────────────────────────────

    def random(self) -> float:
        return self._log("random", (), self._rng.random())

    def gauss(self, mu: float, sigma: float) -> float:
        return self._log("gauss", (mu, sigma), self._rng.gauss(mu, sigma))

    def noisy(self, value: float, sigma: float) -> float:
        """`value` bruité par un gaussien centré. Sigma nul court-circuite le
        tirage : une config déterministe ne doit pas consommer d'aléa."""
        if sigma <= 0:
            return value
        return self._log("noisy", (value, sigma), self._rng.gauss(value, sigma))

    def choice(self, population: Sequence[Any]) -> Any:
        index = self._rng.randrange(len(population))
        self._log("choice", (len(population),), index)
        return population[index]

    def weighted_choice(self, population: Sequence[Any], weights: Sequence[float]) -> Any:
        """Tirage pondéré. Poids tous nuls ou négatifs → uniforme (jamais d'erreur
        au milieu d'une passe)."""
        total = sum(max(0.0, weight) for weight in weights)
        if total <= 0:
            return self.choice(population)
        threshold = self._rng.random() * total
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += max(0.0, weight)
            if threshold < cumulative:
                self._log("weighted_choice", (len(population),), index)
                return population[index]
        self._log("weighted_choice", (len(population),), len(population) - 1)
        return population[-1]

    def softmax_choice(
        self, population: Sequence[Any], scores: Sequence[float], temperature: float
    ) -> Any:
        """Arbitrage stochastique lisible entre options concurrentes (§5).

        Le max est soustrait avant l'exponentielle (stabilité numérique), et une
        température nulle dégénère en argmax déterministe.
        """
        if not population:
            raise ValueError("softmax_choice sur une population vide")
        if temperature <= 0:
            best = max(range(len(scores)), key=lambda index: scores[index])
            self._log("softmax_argmax", (len(population),), best)
            return population[best]
        peak = max(scores)
        weights = [math.exp((score - peak) / temperature) for score in scores]
        return self.weighted_choice(population, weights)

    def shuffled(self, population: Sequence[Any]) -> list[Any]:
        out = list(population)
        self._rng.shuffle(out)
        self._log("shuffled", (len(population),), None)
        return out

    # ── journal ──────────────────────────────────────────────────────────────

    def _log(self, call: str, args: tuple, value: Any) -> Any:
        if self._journal is not None:
            self._journal.append(
                {"key": self.key, "call": call, "args": list(args), "value": value}
            )
        return value


@dataclass
class KeyedRNG:
    """Fabrique de flux. Un flux par `(turn, purpose, faction)`, mémoïsé.

    La mémoïsation est essentielle : deux demandes de la même clé dans le même
    tour doivent POURSUIVRE la même séquence, pas en redémarrer une identique.
    """

    seed: int
    journal_enabled: bool = False
    journal: list[dict] = field(default_factory=list)
    _streams: dict[str, Stream] = field(default_factory=dict, repr=False)

    def stream(self, turn: int, purpose: str, faction: int | None = None) -> Stream:
        key = f"T{turn}:{purpose}" + ("" if faction is None else f":F{faction}")
        existing = self._streams.get(key)
        if existing is not None:
            return existing
        created = Stream(
            key=key,
            _rng=random.Random(derive_seed(self.seed, turn, purpose, faction)),
            _journal=self.journal if self.journal_enabled else None,
        )
        self._streams[key] = created
        return created

    def journal_digest(self) -> str:
        """Empreinte du journal — deux parties identiques ont la même empreinte.

        C'est le test de déterminisme le plus serré : il compare la *trajectoire*
        d'aléa, pas seulement l'état final.
        """
        hasher = hashlib.blake2b(digest_size=16)
        for entry in self.journal:
            hasher.update(
                f"{entry['key']}|{entry['call']}|{entry['args']}|{entry['value']}".encode()
            )
        return hasher.hexdigest()
