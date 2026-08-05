"""Reward de l'environnement.

Les policies LLM ne lisent pas ce signal — elles raisonnent sur le texte. Il existe
pour que l'arène soit réellement entraînable en RL, ce qui est l'objectif affiché du
projet, et parce que PettingZoo l'exige de toute façon.

Deux composantes :

* **shaping** par `Δscore` à chaque passe. Potential-based : la somme des deltas
  télescope vers le score final, donc le shaping ne change pas l'ordre des politiques
  optimales — il ne fait qu'accélérer l'apprentissage.
* **terminal** par rang, pour que le signal soit celui du benchmark (finir premier),
  pas celui de l'accumulation brute.
"""

from __future__ import annotations

from controla.dynamics.scoring import ranks, score
from controla.state.game_state import GameState


def score_snapshot(cfg, state: GameState) -> dict[int, float]:
    return {faction.id: score(cfg, state, faction.id) for faction in state.factions}


def shaping_rewards(cfg, before: dict[int, float], after: dict[int, float]) -> dict[int, float]:
    scale = float(cfg.reward.score_delta_scale)
    return {
        faction_id: (after[faction_id] - before.get(faction_id, 0.0)) * scale
        for faction_id in after
    }


def terminal_rewards(cfg, state: GameState) -> dict[int, float]:
    """Bonus de fin, par rang, plus une pénalité d'élimination.

    Les deux se cumulent : être éliminé ET finir dernier est le pire résultat, ce
    qui est exactement ce qu'on veut décourager.
    """
    bonuses = list(cfg.reward.rank_bonus)
    penalty = float(cfg.reward.elimination_penalty)
    out: dict[int, float] = {}
    for faction_id, rank in ranks(cfg, state).items():
        value = float(bonuses[rank]) if rank < len(bonuses) else float(bonuses[-1])
        if not state.faction(faction_id).alive:
            value += penalty
        out[faction_id] = value
    return out
