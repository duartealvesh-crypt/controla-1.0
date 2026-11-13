"""Le méta-jeu : chaque stratégie complète devient une action pure.

**Table des gains (EGTA).** Le jeu est symétrique une fois les sièges tournés :
une case est un *profil* — le multi-ensemble des 4 stratégies présentes — et
contient le gain moyen (points de rang) de chaque stratégie présente.
Nombre de profils pour K stratégies : C(K + 3, 4).

**Gain contre un mélange.** Les 3 adversaires sont tirés indépendamment selon x :

    f_i(x) = Σ_trios P(trio | x) · gain de i dans le profil {i} ∪ trio

**Équilibre.** Dynamique du réplicateur, à petit pas η :
x_i ← x_i + η · x_i · (f_i(x) − x·f(x)). Ses points stables sont des équilibres
de Nash du méta-jeu symétrique ; si le jeu cycle, on prend la moyenne de la
trajectoire.

**Écart à l'équilibre.** Le jeu des points de rang est à somme constante (6 points
par partie) et symétrique : contre un mélange x joué par tous, on gagne 1,5 en
moyenne. `regret(x) = max_i f_i(x) − x·f(x)` mesure ce qu'une stratégie de la
population gagnerait en déviant ; 0 à l'équilibre (restreint à la population).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations_with_replacement
from math import factorial

import numpy as np

SEATS = 4


def profiles(k: int) -> list[tuple[int, ...]]:
    return list(combinations_with_replacement(range(k), SEATS))


def seatings(profile: tuple[int, ...]) -> list[list[int]]:
    """Les 4 rotations d'un profil : chaque stratégie occupe chaque coin."""
    order = list(profile)
    return [order[r:] + order[:r] for r in range(SEATS)]


def _clean(x) -> np.ndarray:
    x = np.where(x < 1e-6, 0.0, x)
    return x / x.sum()


class PayoffTable:
    def __init__(self):
        # profil → stratégie → [somme des points, nombre de sièges]
        self._cells: dict[tuple, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0])
        )

    def __contains__(self, profile) -> bool:
        return tuple(sorted(profile)) in self._cells

    def missing(self, k: int) -> list[tuple[int, ...]]:
        return [p for p in profiles(k) if p not in self]

    def record(self, seats: list[int], points: list[int]) -> None:
        cell = self._cells[tuple(sorted(seats))]
        for strategy, value in zip(seats, points):
            cell[strategy][0] += value
            cell[strategy][1] += 1

    def payoff(self, profile, strategy: int) -> float:
        total, count = self._cells[tuple(sorted(profile))][strategy]
        return total / count

    # ── analyse ─────────────────────────────────────────────────────────────

    def matrices(self, k: int):
        """A[i, t] = gain de i face au trio t ; C[t, j] = multiplicité de j dans t ;
        coef[t] = coefficient multinomial du trio."""
        trios = list(combinations_with_replacement(range(k), SEATS - 1))
        counts = np.zeros((len(trios), k))
        coef = np.zeros(len(trios))
        payoff = np.zeros((k, len(trios)))
        for t, trio in enumerate(trios):
            for j in trio:
                counts[t, j] += 1
            coef[t] = factorial(SEATS - 1) / np.prod([factorial(int(c)) for c in counts[t]])
            for i in range(k):
                payoff[i, t] = self.payoff((i, *trio), i)
        return payoff, counts, coef

    def fitness(self, x, mats=None) -> np.ndarray:
        payoff, counts, coef = mats if mats is not None else self.matrices(len(x))
        x = np.asarray(x, dtype=float)
        # P(trio) = coef · Π x_j^{c_j} ; 0^0 = 1 traité via c_j = 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(counts > 0, counts * np.log(x), 0.0).sum(axis=1)
        prob = coef * np.exp(log_p)
        return payoff @ prob

    def replicator(self, k: int, x0=None, iterations: int = 20000, step: float = 0.2,
                   tol: float = 1e-10):
        mats = self.matrices(k)
        x = np.full(k, 1.0 / k) if x0 is None else np.asarray(x0, dtype=float)
        average, averaged = np.zeros(k), 0
        for t in range(iterations):
            f = self.fitness(x, mats)
            new = np.clip(x + step * x * (f - x @ f), 0.0, None)
            new /= new.sum()
            if t >= iterations // 2:
                average += new
                averaged += 1
            if np.abs(new - x).max() < tol:
                return _clean(new)
            x = new
        # Pas de convergence : le jeu cycle (pierre-feuille-ciseaux). Le point
        # courant tourne alors autour de l'équilibre, mais la MOYENNE de la
        # trajectoire s'en approche — on garde celui des deux qui regrette le moins.
        candidates = [_clean(x), _clean(average / max(1, averaged))]
        return min(candidates, key=self.regret)

    def regret(self, x) -> float:
        f = self.fitness(x)
        return float(f.max() - np.asarray(x) @ f)

    def to_json(self) -> list:
        return [
            {"profile": list(profile),
             "cells": {str(s): {"sum": v[0], "n": v[1]} for s, v in cells.items()}}
            for profile, cells in self._cells.items()
        ]

    @classmethod
    def from_json(cls, data: list) -> "PayoffTable":
        table = cls()
        for row in data:
            cell = table._cells[tuple(row["profile"])]
            for strategy, value in row["cells"].items():
                cell[int(strategy)] = [value["sum"], value["n"]]
        return table
