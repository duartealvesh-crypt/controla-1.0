"""La campagne : rotations de sièges × seeds × répétitions.

Deux confusions à écarter avant de pouvoir comparer deux modèles :

* **la position.** Les quatre coins sont symétriques par rotation (`world/spawn.py`
  le garantit et un test l'exige), mais les voisinages ne le sont pas : le dieu du
  coin nord-ouest a le même monde que les autres, pas les mêmes vis-à-vis. Faire
  tourner les sièges donne donc à chaque participant chaque coin exactement une
  fois par seed.
* **le monde.** Une seed, c'est un tirage de carte et d'aléa. Comparer sur une seule
  reviendrait à classer des modèles sur une partie de dés.

Le nombre de parties est `seeds × rotations × répétitions`, et le tout est
**déterministe** : rejouer la campagne avec un panel scripté redonne exactement le
même classement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from controla.arena.metrics import Campaign, Criteria
from controla.arena.play import MatchSpec, play_match
from controla.arena.rating import Rating
from controla.arena.storage import write_game, write_manifest
from controla.config.loader import load_config
from controla.policies.factory import build_policies
from controla.registry.catalog import build_catalog
from controla.registry.descriptors import ActionKind


@dataclass
class TournamentSpec:
    """Une campagne : qui joue, sur quelles seeds, avec quelle rotation."""

    seats: list[str]
    seeds: list[int]
    rotations: int = 0          # 0 = autant que de sièges (rotation complète)
    repetitions: int = 1
    run_dir: Path | None = None
    config_path: str | None = None
    # Un JSONL de partie pèse ~125 ko : un sweep de dix-huit tournois en écrirait
    # une centaine de mégaoctets pour n'en lire qu'un tableau de nombres. Le sweep
    # les jette donc ; une campagne de classement les garde, ce sont ses replays.
    keep_games: bool = True
    metadata: dict = field(default_factory=dict)

    def line_ups(self) -> list[tuple[int, list[str]]]:
        """Les alignements de la campagne : `(indice de rotation, sièges)`."""
        count = self.rotations or len(self.seats)
        count = max(1, min(count, len(self.seats)))
        return [
            (index, self.seats[index:] + self.seats[:index])
            for index in range(count)
        ]

    def games(self) -> list[dict]:
        """Le plan de campagne, dans son ordre d'exécution — déterministe."""
        plan = []
        for seed in self.seeds:
            for rotation, line_up in self.line_ups():
                for repetition in range(max(1, self.repetitions)):
                    plan.append({
                        "seed": int(seed),
                        "rotation": rotation,
                        "repetition": repetition,
                        "seats": line_up,
                    })
        return plan

    def to_dict(self) -> dict:
        return {
            "seats": list(self.seats),
            "seeds": [int(seed) for seed in self.seeds],
            "rotations": self.rotations or len(self.seats),
            "repetitions": max(1, self.repetitions),
            "config_path": self.config_path,
            "keep_games": self.keep_games,
            **self.metadata,
        }


def action_universe(catalog) -> int:
    """Le nombre d'actions offertes au dieu — le dénominateur de la cohérence.

    `end_turn` en est exclu : rendre la main n'est pas une intention stratégique.
    """
    return sum(1 for spec in catalog.actions if spec.kind is not ActionKind.CONTROL)


def run_tournament(spec: TournamentSpec, config=None, on_game=None) -> dict:
    """Joue la campagne, écrit un JSONL par partie, renvoie le manifeste.

    `on_game(index, total, result)` est appelé après chaque partie : c'est par là
    que le CLI affiche sa progression, sans que l'arène connaisse la sortie standard.
    """
    cfg = config if config is not None else load_config(spec.config_path)
    catalog = build_catalog(cfg)
    universe = action_universe(catalog)

    directory = Path(spec.run_dir) if spec.run_dir else _default_run_dir()
    directory.mkdir(parents=True, exist_ok=True)

    rating = Rating()
    campaign = Campaign()
    criteria = Criteria()
    max_turns = int(cfg.world.max_turns)
    # Partagé sur toute la campagne : un seul client par modèle, et des statistiques
    # d'appel qui s'accumulent au lieu d'être remises à zéro à chaque partie.
    backends: dict = {}

    plan = spec.games()
    started = time.perf_counter()
    files: list[str] = []

    for index, entry in enumerate(plan):
        match = MatchSpec(
            seed=entry["seed"],
            seats=list(entry["seats"]),
            config_path=spec.config_path,
            metadata={"rotation": entry["rotation"],
                      "repetition": entry["repetition"]},
        )
        result = play_match(
            match, build_policies(match.seats, cfg, catalog, backends), config=cfg,
        )
        rating.observe_match(result)
        campaign.observe_match(result)
        criteria.observe_match(result, max_turns)

        if spec.keep_games:
            path = write_game(directory / f"game_{index:04d}.jsonl", result)
            files.append(path.name)
        if on_game is not None:
            on_game(index, len(plan), result)

    manifest = {
        "run": directory.name,
        "spec": spec.to_dict(),
        "games": len(plan),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "leaderboard": rating.to_dict(),
        "criteria": criteria.report(cfg, rating.standings()),
        "metrics": campaign.report(universe),
        "files": files,
    }
    write_manifest(directory, manifest)
    return manifest


def _default_run_dir() -> Path:
    return Path("runs") / time.strftime("%Y%m%d-%H%M%S")
