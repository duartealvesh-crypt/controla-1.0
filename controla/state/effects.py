"""File d'effets datés (spec §11).

Toute action à durée passe par ici : chantiers, armées en marche, tributs, réponses
de négociation. `latency` / `duration` sont des propriétés génériques — aucun cas
particulier par action.

Deux choix qui simplifient beaucoup :

* **Tour absolu de résolution** (`due_turn`), pas un compte à rebours. Moins de
  bugs, et la sérialisation/replay devient triviale (rien à décrémenter).
* **Ordre de résolution stable** : `(due_turn, seq)`, où `seq` est un compteur
  d'insertion monotone. Deux effets échus au même tour se résolvent toujours dans
  le même ordre, donc le replay est fidèle.

La file d'un dieu est visible dans SON information_state (« ton armée arrive dans
1 tour ») et cachée chez les autres — espionner, c'est lire la file adverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Modifier:
    """Modificateur multiplicatif daté, porté par une case OU par une faction.

    `amount` est un écart relatif (+0.70 = +70 %, -0.60 = -60 %). Il expire quand
    `turn > until_turn` — tour absolu, jamais un compte à rebours.

    Le même type sert aux deux échelles : une bénédiction sur une case et une
    maladie sur une natalité sont la même mécanique, donc pas de cas particulier.
    """

    amount: float
    until_turn: int
    cause: str = ""

    def to_dict(self) -> dict:
        return {"amount": self.amount, "until_turn": self.until_turn, "cause": self.cause}


def active_multiplier(modifiers: dict[str, Modifier], turn: int) -> float:
    """Produit des modificateurs actifs, borné à 0 : rien ne devient négatif, même
    sous plusieurs malus cumulés."""
    multiplier = 1.0
    for modifier in modifiers.values():
        if modifier.until_turn >= turn:
            multiplier *= 1.0 + modifier.amount
    return max(0.0, multiplier)


def expire(modifiers: dict[str, Modifier], turn: int) -> list[str]:
    """Retire les modificateurs échus. Renvoie les clés retirées (pour le log)."""
    expired = [key for key, mod in modifiers.items() if mod.until_turn < turn]
    for key in expired:
        del modifiers[key]
    return expired


@dataclass
class PendingEffect:
    """Un effet qui se résoudra à `due_turn`.

    `cause` est du texte lisible : il alimente l'attribution (« Bénédiction
    (plaine nord) : +12 nourriture ») pour que le dieu voie sa causalité même
    médiée (§8).
    """

    kind: str
    due_turn: int
    actor: int
    cause: str
    payload: dict = field(default_factory=dict)
    seq: int = 0
    cancelled: bool = False

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "due_turn": self.due_turn,
            "actor": self.actor,
            "cause": self.cause,
            "payload": dict(self.payload),
            "seq": self.seq,
            "cancelled": self.cancelled,
        }


class EffectQueue:
    """File d'effets datés, à ordre de résolution déterministe."""

    __slots__ = ("_effects", "_next_seq")

    def __init__(self) -> None:
        self._effects: list[PendingEffect] = []
        self._next_seq = 0

    def __len__(self) -> int:
        return len(self._effects)

    def add(
        self,
        kind: str,
        due_turn: int,
        actor: int,
        cause: str = "",
        **payload,
    ) -> PendingEffect:
        effect = PendingEffect(
            kind=kind,
            due_turn=due_turn,
            actor=actor,
            cause=cause,
            payload=payload,
            seq=self._next_seq,
        )
        self._next_seq += 1
        self._effects.append(effect)
        return effect

    def pop_due(self, turn: int) -> list[PendingEffect]:
        """Retire et renvoie les effets échus, dans l'ordre `(due_turn, seq)`.

        Les effets annulés sont retirés aussi mais pas renvoyés : une armée en
        production dans une ville rasée avant l'échéance disparaît sans se résoudre.
        """
        due = [effect for effect in self._effects if effect.due_turn <= turn]
        self._effects = [effect for effect in self._effects if effect.due_turn > turn]
        due.sort(key=lambda effect: (effect.due_turn, effect.seq))
        return [effect for effect in due if not effect.cancelled]

    def pending(self, actor: int | None = None, kind: str | None = None) -> list[PendingEffect]:
        """Effets encore en attente, filtrables. Ordre stable."""
        out = [
            effect
            for effect in self._effects
            if not effect.cancelled
            and (actor is None or effect.actor == actor)
            and (kind is None or effect.kind == kind)
        ]
        out.sort(key=lambda effect: (effect.due_turn, effect.seq))
        return out

    def cancel(self, kind: str | None = None, actor: int | None = None, **match) -> int:
        """Annule les effets correspondants. Renvoie le nombre d'annulations.

        Les critères de `match` sont comparés au payload — d'où l'annulation
        ciblée (« l'armée qui partait de cette ville-là »).
        """
        count = 0
        for effect in self._effects:
            if effect.cancelled:
                continue
            if kind is not None and effect.kind != kind:
                continue
            if actor is not None and effect.actor != actor:
                continue
            if any(effect.payload.get(key) != value for key, value in match.items()):
                continue
            effect.cancelled = True
            count += 1
        return count

    def to_dict(self) -> dict:
        return {
            "next_seq": self._next_seq,
            "effects": [effect.to_dict() for effect in self._effects],
        }
