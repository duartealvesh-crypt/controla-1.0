"""L'interface de policy : `obs → action`, et rien d'autre.

Les trois couches sont étanches (§11) :

| Couche      | Responsabilité                                              |
|-------------|-------------------------------------------------------------|
| Environment | règles, dynamique, reward, légalité, seeding, observations   |
| Policy      | `obs → action` uniquement (LLM, scriptée, ou RL)             |
| Arena       | assigne les policies aux sièges, joue, calcule les ratings   |

L'environnement **ignore** qui occupe un siège : c'est ce qui permet de comparer un
LLM, un archétype scripté et un agent RL sans qu'aucun ne bénéficie d'un traitement
particulier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Policy(ABC):
    """Décide d'un coup à partir d'une observation.

    Un coup est un **index** dans l'espace d'action, choisi parmi ceux dont le
    masque vaut 1 — sinon il sera compté invalide (métrique de qualité
    d'instruction-following) et perdu.

    Une policy textuelle peut renvoyer `{"action": int, "argument": str}` pour
    joindre un argumentaire de négociation ; c'est le seul écart au contrat RL.
    """

    name: str = "policy"

    def reset(self, faction_id: int, seed: int) -> None:
        """Appelée une fois par partie, avant le premier coup."""

    @abstractmethod
    def act(self, observation: dict, info: dict) -> dict:
        """Renvoie un coup. `info["view"]` porte la vue structurée lisible."""

    def observe_turn(self, report: dict) -> None:
        """Notifiée après chaque passe — permet de tenir un historique."""


def action(index: int, argument: str = "") -> int | dict:
    """Construit un coup bien formé.

    Renvoie un `int` nu quand il n'y a pas d'argumentaire, pour rester exactement
    sur le contrat `Discrete` déclaré par l'environnement.
    """
    if argument:
        return {"action": int(index), "argument": argument}
    return int(index)
