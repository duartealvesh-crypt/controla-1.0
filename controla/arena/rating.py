"""Classement des participants : OpenSkill / Plackett-Luce, ordinal `μ − 3σ`.

Pourquoi Plackett-Luce plutôt qu'un Elo ou un TrueSkill classique : une partie de
Controla ne produit pas un vainqueur mais un **classement complet des quatre dieux**
(§10 garantit l'absence de match nul). Plackett-Luce est exactement le modèle de ce
signal — il exploite les trois comparaisons d'un classement à quatre là où un
« gagnant / perdants » en jetterait deux.

L'ordinal `μ − 3σ` est un classement **prudent** : un modèle n'y monte qu'après avoir
prouvé sa force sur assez de parties. Une pointe de chance sur trois parties laisse
`σ` élevé, donc l'ordinal bas. C'est ce qu'on veut d'un benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openskill.models import PlackettLuce


@dataclass
class Standing:
    """Ce qu'une campagne sait d'un participant."""

    name: str
    mu: float
    sigma: float
    ordinal: float
    games: int = 0
    wins: int = 0
    rank_sum: int = 0
    seats: int = 0          # sièges occupés (≥ games si le participant en tient deux)

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.seats if self.seats else None

    @property
    def mean_rank(self) -> float | None:
        return self.rank_sum / self.seats if self.seats else None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ordinal": round(self.ordinal, 3),
            "mu": round(self.mu, 3),
            "sigma": round(self.sigma, 3),
            "games": self.games,
            "seats": self.seats,
            "wins": self.wins,
            "win_rate": None if self.win_rate is None else round(self.win_rate, 4),
            "mean_rank": None if self.mean_rank is None else round(self.mean_rank, 3),
        }


@dataclass
class Rating:
    """Classement mis à jour partie par partie.

    L'ordre de traitement des parties compte (les mises à jour sont séquentielles) ;
    le tournoi les joue dans un ordre déterministe, donc une campagne rejouée donne
    exactement le même classement.
    """

    model: PlackettLuce = field(default_factory=PlackettLuce)
    ratings: dict = field(default_factory=dict)
    tally: dict[str, list[int]] = field(default_factory=dict)   # [seats, wins, rank_sum]
    games: dict[str, int] = field(default_factory=dict)

    def _rating(self, name: str):
        if name not in self.ratings:
            self.ratings[name] = self.model.rating(name=name)
        return self.ratings[name]

    def observe(self, seats: list[str], ranks: list[int]) -> None:
        """Enregistre une partie. `ranks[i]` = rang du siège `i` (0 = vainqueur)."""
        teams = [[self._rating(name)] for name in seats]
        updated = self.model.rate(teams, ranks=list(ranks))

        # Un participant peut occuper plusieurs sièges d'une même partie (typiquement
        # la ligne de base répétée : `a,b,heuristic,heuristic`). Ses postérieurs sont
        # alors **moyennés** — sans quoi le dernier siège traité écraserait les autres
        # et le participant serait jugé sur un seul de ses résultats, tiré au hasard
        # par l'ordre des sièges.
        #
        # Limite mesurée, à ne pas maquiller : σ, lui, décroît un peu plus vite pour
        # un participant qui occupe deux sièges, parce que la partie contient
        # réellement plus de comparaisons le concernant. Six parties sur deux sièges
        # valent à peu près douze parties sur un siège (σ 6.50 contre 6.67) alors
        # qu'elles devraient valoir moins : la corrélation entre deux sièges d'un
        # même monde n'est pas modélisée par OpenSkill. C'est une surconfiance de
        # ~2 % qui frappe la ligne de base, jamais les modèles comparés — la garder
        # hors du classement de tête est la bonne lecture.
        posteriors: dict[str, list] = {}
        for name, team in zip(seats, updated):
            posteriors.setdefault(name, []).append(team[0])
        for name, values in posteriors.items():
            self.ratings[name] = self.model.rating(
                mu=sum(value.mu for value in values) / len(values),
                sigma=sum(value.sigma for value in values) / len(values),
                name=name,
            )
            self.games[name] = self.games.get(name, 0) + 1

        for name, rank in zip(seats, ranks):
            counters = self.tally.setdefault(name, [0, 0, 0])
            counters[0] += 1
            counters[1] += int(rank == 0)
            counters[2] += int(rank)

    def observe_match(self, result: dict) -> None:
        """Enregistre une partie depuis le résultat renvoyé par `play_match`."""
        seats = list(result["seats"])
        ranks = [0] * len(seats)
        for entry in result["ranking"]:
            ranks[int(entry["faction_id"])] = int(entry["rank"])
        self.observe(seats, ranks)

    def standings(self) -> list[Standing]:
        """Les participants, du plus fort au plus faible selon `μ − 3σ`."""
        table = [
            Standing(
                name=name,
                mu=rating.mu,
                sigma=rating.sigma,
                ordinal=rating.ordinal(),
                games=self.games.get(name, 0),
                seats=self.tally.get(name, [0, 0, 0])[0],
                wins=self.tally.get(name, [0, 0, 0])[1],
                rank_sum=self.tally.get(name, [0, 0, 0])[2],
            )
            for name, rating in self.ratings.items()
        ]
        # Le nom départage : deux participants jamais départagés par les parties
        # doivent quand même sortir dans un ordre stable d'une exécution à l'autre.
        return sorted(table, key=lambda entry: (-entry.ordinal, entry.name))

    def to_dict(self) -> list[dict]:
        return [standing.to_dict() for standing in self.standings()]
