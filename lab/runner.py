"""Jouer beaucoup de parties en parallèle, et n'en garder que les points de rang.

Un `Runner` garde un seul pool de processus pour toute une exécution : sous
Windows, lancer un processus coûte environ une seconde, ce qu'on ne veut pas payer
à chaque génération de CMA-ES.

Les processus de calcul tournent en **priorité basse** : ils prennent tout le CPU
libre, mais le cèdent dès que l'utilisateur se sert de sa machine (mesuré : en
priorité normale, 11 processus rendaient le PC inutilisable pendant 2 h).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

from controla import api

RANK_POINTS = (3, 2, 1, 0)


def points_of(seats: list, seed: int, overrides: dict | None = None) -> list[int]:
    """Points de rang de chaque siège (dans l'ordre des sièges) pour une partie."""
    result = api.play_game(seats, seed, overrides)
    points = [0] * len(seats)
    for item in result["ranking"]:
        points[item["faction_id"]] = RANK_POINTS[item["rank"]]
    return points


def lower_priority() -> None:
    """Passe le processus courant en priorité « inférieure à la normale »."""
    try:
        if os.name == "nt":
            import ctypes

            below_normal = 0x4000
            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal)
        else:
            os.nice(10)
    except (OSError, AttributeError):
        pass  # simple politesse : ne jamais faire échouer un calcul pour elle


def _job(job: tuple) -> list[int]:
    seats, seed, overrides = job
    return points_of(seats, seed, overrides)


class Runner:
    def __init__(self, workers: int | None = None, overrides: dict | None = None,
                 low_priority: bool = True):
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        self.overrides = overrides
        self.low_priority = low_priority
        self._pool: ProcessPoolExecutor | None = None
        self.games = 0

    def __enter__(self) -> "Runner":
        if self.workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=lower_priority if self.low_priority else None,
            )
        return self

    def __exit__(self, *exc) -> None:
        if self._pool is not None:
            self._pool.shutdown()

    def play(self, games: list[tuple[list, int]]) -> list[list[int]]:
        """`games` : liste de (sièges, seed). Rend les points de chaque partie."""
        jobs = [(seats, seed, self.overrides) for seats, seed in games]
        self.games += len(jobs)
        if self._pool is None:
            return [_job(job) for job in jobs]
        chunk = max(1, len(jobs) // (self.workers * 4))
        return list(self._pool.map(_job, jobs, chunksize=chunk))
