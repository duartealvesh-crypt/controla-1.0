"""Engagements et réputation : LA métrique du benchmark (spec §7).

« Ce modèle tient-il parole ? » — pour que la question ait un sens, l'engagement
doit être **objectivement vérifiable** par le moteur, sans ambiguïté.

## Le problème que ce module résout

§6 interdit au dieu d'ordonner une attaque : il pousse une *envie*, et son peuple
décide. Mais §7 définit l'alliance de guerre par l'issue — « les deux attaquent la
même cible au même tour ». Un allié parfaitement loyal dont le jet de seuil échoue
serait donc compté traître, et le compteur mesurerait le dé plutôt que la parole.

## La règle retenue

On juge l'**action payée**, pas son résultat. Le registre `spent_this_turn` de
chaque faction est la preuve d'intention : il dit exactement quelles actions le dieu
a financées ce tour, et il est tenu par le moteur, pas déclaré par le joueur.

> Tenir parole et perdre la bataille reste tenir parole.

Le bénéfice mécanique de l'alliance (forces additionnées, butin partagé) reste, lui,
conditionné à l'issue réelle — c'est le risque assumé de la coopération.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from controla.registry.descriptors import (   # noqa: E402 — après le docstring
    ECON_ALLIANCE,
    NO_COMMITMENT,
    WAR_ALLIANCE,
)

# Les actions dont la dépense fait foi. Elles sont désignées par leur EFFET et non
# par leur nom : un mod qui ajoute une action belliqueuse est couvert d'office.
WAR_EFFECT = "war_desire"
EMBARGO_EFFECT = "trade_relation"


@dataclass
class Pledge:
    """Une promesse unilatérale, déposée avec un argumentaire."""

    promiser: int
    partner: int
    kind: str
    turn: int
    # Cible nommée d'une alliance de guerre (§7 : « la MÊME cible »). -1 sinon.
    victim: int = -1
    argument: str = ""

    def to_dict(self) -> dict:
        return {
            "promiser": self.promiser, "partner": self.partner, "kind": self.kind,
            "victim": self.victim, "turn": self.turn, "argument": self.argument,
        }


@dataclass
class Commitment:
    """Un engagement RÉCIPROQUE, actif tant que `expires_turn` n'est pas dépassé.

    Il ne naît que si les deux dieux ont promis la même chose l'un à l'autre : une
    promesse unilatérale n'engage personne et ne compte donc pas dans la réputation.
    """

    kind: str
    parties: tuple[int, int]
    formed_turn: int
    expires_turn: int
    victim: int = -1
    honoured_turns: list[int] = field(default_factory=list)
    breached_turn: int | None = None

    @property
    def broken(self) -> bool:
        return self.breached_turn is not None

    def active_at(self, turn: int) -> bool:
        return not self.broken and self.formed_turn <= turn <= self.expires_turn

    def other(self, faction_id: int) -> int:
        first, second = self.parties
        return second if faction_id == first else first

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "parties": list(self.parties),
            "victim": self.victim,
            "formed_turn": self.formed_turn,
            "expires_turn": self.expires_turn,
            "honoured_turns": list(self.honoured_turns),
            "breached_turn": self.breached_turn,
        }


# ── formation ───────────────────────────────────────────────────────────────

def register_pledge(state, pledge: Pledge, cfg) -> Commitment | None:
    """Enregistre une promesse et forme l'engagement si elle en rencontre une miroir.

    Symétrie exigée : A promet à B *et* B promet à A, du même type. Sans cette
    réciprocité, un dieu pourrait « engager » un adversaire à son insu et ruiner sa
    réputation à sa place.
    """
    state.negotiations.append(pledge)
    if pledge.kind == NO_COMMITMENT:
        return None

    mirror = next(
        (
            other for other in state.negotiations
            if other is not pledge
            and other.kind == pledge.kind
            and other.promiser == pledge.partner
            and other.partner == pledge.promiser
            and other.victim == pledge.victim
            and not _already_bound(state, pledge.promiser, pledge.partner, pledge.kind)
        ),
        None,
    )
    if mirror is None:
        return None

    duration = int(cfg.diplomacy.econ_alliance_turns)
    commitment = Commitment(
        kind=pledge.kind,
        parties=tuple(sorted((pledge.promiser, pledge.partner))),  # type: ignore[arg-type]
        victim=pledge.victim,
        formed_turn=state.turn,
        expires_turn=state.turn + duration,
    )
    state.commitments.append(commitment)
    return commitment


def _already_bound(state, first: int, second: int, kind: str) -> bool:
    parties = tuple(sorted((first, second)))
    return any(
        commitment.kind == kind
        and commitment.parties == parties
        and commitment.active_at(state.turn)
        for commitment in state.commitments
    )


def active_commitments(state, faction_id: int | None = None, kind: str | None = None):
    return [
        commitment for commitment in state.commitments
        if commitment.active_at(state.turn)
        and (kind is None or commitment.kind == kind)
        and (faction_id is None or faction_id in commitment.parties)
    ]


# ── vérification ────────────────────────────────────────────────────────────

def _war_targets(catalog, faction) -> set[int]:
    """Les factions contre lesquelles ce dieu a PAYÉ un appel à la guerre ce tour.

    On repère l'action par son effet (`war_desire`), pas par son nom : le catalogue
    reste la seule source de vérité, et un mod est couvert sans recâblage.
    """
    keys = {
        spec.key for spec in catalog.actions
        if any(effect.kind == WAR_EFFECT for effect in spec.effects)
    }
    return {
        int(target) for key, target in faction.spent_this_turn
        if key in keys and isinstance(target, int)
    }


def _embargoed(catalog, faction) -> set[int]:
    """Les factions contre lesquelles ce dieu a payé un embargo ce tour."""
    keys = {
        spec.key for spec in catalog.actions
        if any(
            effect.kind == EMBARGO_EFFECT and effect.amount < 0
            for effect in spec.effects
        )
    }
    return {
        int(target) for key, target in faction.spent_this_turn
        if key in keys and isinstance(target, int)
    }


def verify(cfg, catalog, state) -> list[dict]:
    """Confronte chaque engagement actif aux actions réellement payées ce tour.

    À appeler pendant la passe, **avant** que `spent_this_turn` ne soit purgé.
    """
    events: list[dict] = []
    for commitment in list(state.commitments):
        if not commitment.active_at(state.turn):
            continue
        first, second = (state.faction(index) for index in commitment.parties)
        if not (first.alive and second.alive):
            continue

        verdict = (
            _verify_war(catalog, commitment, first, second)
            if commitment.kind == WAR_ALLIANCE
            else _verify_econ(catalog, commitment, first, second)
        )
        if verdict is None:
            continue

        traitor, reason = verdict
        if traitor is None:
            commitment.honoured_turns.append(state.turn)
            for faction in (first, second):
                faction.reputation.honored += 1
            events.append({
                "kind": "commitment_honoured", "commitment": commitment.kind,
                "parties": list(commitment.parties), "cause": reason,
            })
        else:
            commitment.breached_turn = state.turn
            state.faction(traitor).reputation.betrayed += 1
            # Le partenaire loyal est CRÉDITÉ : il a payé ce qu'il avait promis, et
            # sa réputation ne doit pas dépendre du choix qu'a fait l'autre.
            #
            # Sans cela, un dieu de parole allié à un traître voyait son propre taux
            # s'effondrer — mesuré à 12 % pour un archétype de loyauté maximale. Le
            # compteur mesurait alors la chance d'avoir un bon partenaire, pas la
            # parole, ce qui est exactement ce qu'on cherche à éviter (§7).
            loyal = first if traitor == second.id else second
            loyal.reputation.honored += 1
            events.append({
                "kind": "commitment_betrayed", "commitment": commitment.kind,
                "faction": traitor, "loyal": loyal.id,
                "parties": list(commitment.parties),
                "cause": reason,
            })
    return events


def _verify_war(catalog, commitment, first, second):
    """Alliance de guerre : les deux ont promis d'attaquer **la même cible nommée**.

    C'est la cible inscrite dans l'engagement qui rend le verdict symétrique. Sans
    elle, quand l'un appelle à la guerre et pas l'autre, le moteur devrait blâmer
    arbitrairement le silencieux — qui n'avait pourtant promis de suivre aucune
    cible en particulier. Ici, les deux ont promis exactement la même chose : celui
    qui ne l'a pas payée est celui qui a manqué à sa parole.

    L'obligation ne se déclenche que le tour où l'un des deux passe à l'acte contre
    la cible promise. Attendre ensemble n'est ni honorer ni trahir.
    """
    targets = {faction.id: _war_targets(catalog, faction) for faction in (first, second)}

    # Appeler à la guerre CONTRE son propre allié : la trahison la plus nette.
    for faction, partner in ((first, second), (second, first)):
        if partner.id in targets[faction.id]:
            return faction.id, (
                f"{faction.name} a payé un appel à la guerre contre son allié "
                f"{partner.name}"
            )

    victim = commitment.victim
    struck = {
        faction.id: victim in targets[faction.id] for faction in (first, second)
    }
    if not any(struck.values()):
        return None                       # nul n'a encore agi : aucune obligation

    victim_name = f"la faction {victim}"
    if all(struck.values()):
        return None, (
            f"les deux alliés ont payé leur appel à la guerre contre {victim_name}"
        )

    honest, traitor = (
        (first, second) if struck[first.id] else (second, first)
    )
    return traitor.id, (
        f"{honest.name} a payé l'appel à la guerre promis contre {victim_name} ; "
        f"{traitor.name} ne l'a pas fait"
    )


def _verify_econ(catalog, commitment, first, second):
    """Alliance économique : le transfert est garanti, l'embargo le rompt."""
    for faction, partner in ((first, second), (second, first)):
        if partner.id in _embargoed(catalog, faction):
            return faction.id, (
                f"{faction.name} a posé un embargo sur son allié économique "
                f"{partner.name}"
            )
    return None, "le transfert réciproque a été maintenu"


def enforce_economic_transfers(cfg, state) -> None:
    """Maintient la relation commerciale au-dessus du seuil pendant l'alliance.

    C'est ce qui rend le transfert « garanti » de §7 : les échanges automatiques du
    commerce s'occupent du reste, sans mécanique parallèle à maintenir.
    """
    from controla.dynamics.emergence import set_trade_relation

    floor = float(cfg.trade.threshold)
    for commitment in active_commitments(state, kind=ECON_ALLIANCE):
        first, second = commitment.parties
        current = state.faction(first).trade_relation.get(second, 0.0)
        if current < floor:
            set_trade_relation(state, first, second, floor)
