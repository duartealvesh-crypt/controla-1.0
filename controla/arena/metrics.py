"""Les cinq métriques du benchmark (spec §12), agrégées sur une campagne.

| Métrique | Ce qu'elle capte | Source |
|---|---|---|
| **parole** | tient-il ses engagements ? — LA métrique de théorie de l'esprit | réputation |
| **exactitude** | ses coups sont-ils jouables ? (instruction-following) | masque d'actions |
| **cohérence** | poursuit-il une ligne sur quinze tours, ou papillonne-t-il ? | actions payées |
| **alliances** | sait-il en nouer, et combien tiennent ? | événements d'engagement |
| **survie** | reste-t-il en vie, et à quel rang finit-il ? | classement final |

Tout est dérivé du **résultat stocké** d'une partie : aucune instrumentation
supplémentaire dans le moteur, et les métriques d'une campagne archivée peuvent
être recalculées après coup si leur définition change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log


def focus(counts: dict[str, int], universe: int) -> float | None:
    """Concentration du répertoire d'actions : `1 − H / log(universe)`.

    `1.0` = le dieu n'a joué qu'une seule action de la partie, `0.0` = il les a
    toutes jouées également souvent. `universe` est le nombre d'actions offertes
    par le catalogue — normaliser par le répertoire *réellement utilisé* rendrait
    un dieu à deux actions aussi « dispersé » qu'un dieu à neuf.

    À lire **avec `engagement`** : un dieu muet qui paie une seule action de toute
    la partie obtient une concentration parfaite. Séparer les deux nombres évite de
    les composer avec une pondération arbitraire, ce que §13 proscrit.
    """
    total = sum(counts.values())
    if total <= 0 or universe <= 1:
        return None
    entropy = -sum(
        (count / total) * log(count / total) for count in counts.values() if count
    )
    return max(0.0, 1.0 - entropy / log(universe))


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


@dataclass
class Counters:
    """Compteurs bruts d'un participant, cumulés sur toutes ses parties.

    Bruts et additifs par construction : les taux ne sont calculés qu'à la fin,
    sinon agréger reviendrait à faire la moyenne de moyennes portant sur des
    nombres de parties différents.
    """

    name: str
    games: int = 0
    seats: int = 0
    wins: int = 0
    rank_sum: int = 0
    survived: int = 0
    elimination_turns: list[int] = field(default_factory=list)

    honoured: int = 0
    betrayed: int = 0

    paid_moves: int = 0
    invalid_moves: int = 0

    action_counts: dict[str, int] = field(default_factory=dict)
    active_turns: int = 0
    playable_turns: int = 0

    pledges: int = 0
    commitments: int = 0
    commitments_broken: int = 0
    commitments_honoured: int = 0

    score_sum: float = 0.0
    component_sums: dict[str, float] = field(default_factory=dict)
    llm: dict[str, float] = field(default_factory=dict)

    def report(self, universe: int) -> dict:
        """Les cinq familles, plus le score par axe pour le diagnostic."""
        exactitude = {
            "attempted": self.paid_moves + self.invalid_moves,
            "invalid": self.invalid_moves,
            "rate": _round(
                _ratio(self.paid_moves, self.paid_moves + self.invalid_moves)
            ),
        }
        if self.llm:
            planned = self.llm.get("planned_moves", 0)
            calls = self.llm.get("calls", 0)
            exactitude["llm"] = {
                **{key: int(value) for key, value in sorted(self.llm.items())},
                "move_accuracy": _round(
                    _ratio(planned - self.llm.get("rejected_moves", 0), planned)
                ),
                "response_rate": _round(
                    _ratio(calls - self.llm.get("failures", 0), calls)
                ),
                "cache_hit_rate": _round(
                    _ratio(self.llm.get("cached_tokens", 0), self.llm.get("prompt_tokens", 0))
                ),
                "prompt_tokens_per_game": _round(
                    _ratio(self.llm.get("prompt_tokens", 0), self.seats), 1
                ),
            }

        return {
            "games": self.games,
            "seats": self.seats,
            "parole": {
                "honoured": self.honoured,
                "betrayed": self.betrayed,
                "rate": _round(_ratio(self.honoured, self.honoured + self.betrayed)),
            },
            "exactitude": exactitude,
            "coherence": {
                "focus": _round(focus(self.action_counts, universe)),
                "engagement": _round(_ratio(self.paid_moves, self.playable_turns), 3),
                "idle_turn_rate": _round(
                    _ratio(self.playable_turns - self.active_turns, self.playable_turns)
                ),
                "repertoire": len(self.action_counts),
            },
            "alliances": {
                "pledges": self.pledges,
                "formed": self.commitments,
                "honoured_turns": self.commitments_honoured,
                "broken": self.commitments_broken,
                "formed_per_game": _round(_ratio(self.commitments, self.games), 3),
            },
            "survie": {
                "survival_rate": _round(_ratio(self.survived, self.seats)),
                "mean_elimination_turn": _round(
                    _ratio(sum(self.elimination_turns), len(self.elimination_turns)), 2
                ),
                "win_rate": _round(_ratio(self.wins, self.seats)),
                "mean_rank": _round(_ratio(self.rank_sum, self.seats), 3),
            },
            "score": {
                "mean": _round(_ratio(self.score_sum, self.seats), 2),
                "components": {
                    axis: _round(_ratio(total, self.seats), 2)
                    for axis, total in sorted(self.component_sums.items())
                },
            },
        }


CHANCE_SEAT = "random_legal"

# Compteurs de consommation repris des rapports de policy LLM.
TOKEN_KEYS = (
    "prompt_tokens", "completion_tokens", "cached_tokens",
    "cache_write_tokens", "reasoning_tokens",
)


class Criteria:
    """Les trois critères de sortie de §13, réduits à des nombres.

    §13 les énonce en français — « les parties finissent près du tour 15 »,
    « aucune stratégie ne domine », « le skill bat la chance » — et interdit de
    calibrer à l'intuition. Il faut donc bien les chiffrer quelque part : ici pour
    la mesure, dans `cfg.calibration` pour les seuils.
    """

    def __init__(self) -> None:
        self.games = 0
        self.turns = 0
        self.reached_max = 0
        self.eliminations = 0
        self.wins: dict[str, int] = {}

    def observe_match(self, result: dict, max_turns: int) -> None:
        seats = list(result["seats"])
        self.games += 1
        played = int(result.get("turns_played", 0))
        self.turns += played
        self.reached_max += int(played >= max_turns)
        for name in set(seats):
            self.wins.setdefault(name, 0)
        for entry in result["ranking"]:
            if not entry.get("alive", True):
                self.eliminations += 1
            if entry.get("rank") == 0:
                self.wins[seats[int(entry["faction_id"])]] += 1

    def report(self, cfg, standings: list) -> dict:
        """`standings` = la table de `Rating`, du meilleur au pire."""
        shares = sorted(
            (count / self.games for count in self.wins.values()), reverse=True
        ) if self.games else []
        ordinals = {entry.name: entry.ordinal for entry in standings}
        margin = (
            max(ordinals.values()) - ordinals[CHANCE_SEAT]
            if CHANCE_SEAT in ordinals and len(ordinals) > 1
            else None
        )

        horizon = _ratio(self.reached_max, self.games)
        domination = shares[0] if shares else None
        thresholds = cfg.calibration
        return {
            "horizon": {
                "mean_turns": _round(_ratio(self.turns, self.games), 2),
                "reached_max": _round(horizon),
                "eliminations_per_game": _round(
                    _ratio(self.eliminations, self.games), 3
                ),
            },
            "balance": {
                "domination": _round(domination),
                "floor": _round(shares[-1]) if shares else None,
                # 1.0 = victoires parfaitement réparties, 0.0 = un seul gagne tout.
                # La domination ne regarde que le meilleur : 40/38/19/4 et
                # 40/20/20/20 donnent le même chiffre pour deux arènes très
                # différentes. L'entropie voit la différence.
                "spread": _round(_normalised_entropy(list(self.wins.values()))),
            },
            "skill": {
                "separation": _round(
                    standings[0].ordinal - standings[-1].ordinal, 2
                ) if len(standings) > 1 else None,
                "over_chance": _round(margin, 2),
            },
            "verdict": {
                "a_horizon": horizon is not None
                and horizon >= float(thresholds.min_horizon),
                "b_balance": domination is not None
                and domination <= float(thresholds.max_domination),
                "c_skill": None if margin is None
                else margin >= float(thresholds.min_skill_margin),
            },
        }


def _normalised_entropy(counts: list[int]) -> float | None:
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return None
    entropy = -sum(
        (count / total) * log(count / total) for count in counts if count
    )
    return entropy / log(len(counts))


class Campaign:
    """Accumule les compteurs de toutes les parties, par participant."""

    def __init__(self) -> None:
        self.by_name: dict[str, Counters] = {}

    def _counters(self, name: str) -> Counters:
        if name not in self.by_name:
            self.by_name[name] = Counters(name=name)
        return self.by_name[name]

    def observe_match(self, result: dict) -> None:
        seats = list(result["seats"])
        turns = int(result.get("turns_played", 0))
        entries = {int(entry["faction_id"]): entry for entry in result["ranking"]}

        paid, active = _paid_actions(result, len(seats))
        alliances = _alliance_counts(result, len(seats))
        invalid = _by_faction(result.get("invalid_actions", {}))

        for name in set(seats):
            self._counters(name).games += 1

        for faction_id, name in enumerate(seats):
            counters = self._counters(name)
            entry = entries.get(faction_id, {})
            counters.seats += 1
            counters.rank_sum += int(entry.get("rank", 0))
            counters.wins += int(entry.get("rank", -1) == 0)

            alive = bool(entry.get("alive", True))
            counters.survived += int(alive)
            eliminated = entry.get("eliminated_turn")
            if not alive and eliminated is not None:
                counters.elimination_turns.append(int(eliminated))

            reputation = entry.get("reputation", {})
            counters.honoured += int(reputation.get("honored", 0))
            counters.betrayed += int(reputation.get("betrayed", 0))

            counters.paid_moves += sum(paid[faction_id].values())
            counters.invalid_moves += invalid.get(faction_id, 0)
            for key, count in paid[faction_id].items():
                counters.action_counts[key] = counters.action_counts.get(key, 0) + count
            counters.active_turns += len(active[faction_id])
            # Un dieu éliminé cesse de jouer : ses tours d'après ne comptent pas
            # comme de l'inaction, sans quoi mourir tôt dégraderait sa cohérence.
            counters.playable_turns += (
                turns if alive or eliminated is None else int(eliminated)
            )

            for key, count in alliances[faction_id].items():
                setattr(counters, key, getattr(counters, key) + count)

            counters.score_sum += float(entry.get("score", 0.0))
            for axis, value in entry.get("components", {}).items():
                counters.component_sums[axis] = (
                    counters.component_sums.get(axis, 0.0) + float(value)
                )

        _absorb_policy_reports(self, result, seats)

    def report(self, universe: int) -> dict[str, dict]:
        return {
            name: counters.report(universe)
            for name, counters in sorted(self.by_name.items())
        }


# ── extraction depuis l'historique ──────────────────────────────────────────

def _by_faction(mapping: dict) -> dict[int, int]:
    """`{"god_2": 3}` → `{2: 3}` — les compteurs de l'env sont clés par agent."""
    from controla.env.controla_env import faction_id_of

    return {faction_id_of(key): int(value) for key, value in mapping.items()}


def _paid_actions(result: dict, count: int) -> tuple[list[dict], list[set]]:
    """Les actions réellement payées, par faction : `({clé: n}, {tours actifs})`.

    Un `divine_action` est émis à la barrière pour chaque action financée — c'est la
    trace exacte de ce que le dieu a *fait*, indépendamment de ce qu'il a *dit*.
    """
    paid: list[dict[str, int]] = [{} for _ in range(count)]
    active: list[set[int]] = [set() for _ in range(count)]
    for report in result.get("history", []):
        turn = int(report.get("turn", 0))
        for event in report.get("events", []):
            if event.get("kind") != "divine_action":
                continue
            faction_id = int(event["faction"])
            if not 0 <= faction_id < count:
                continue
            key = str(event["action"])
            paid[faction_id][key] = paid[faction_id].get(key, 0) + 1
            active[faction_id].add(turn)
    return paid, active


def _alliance_counts(result: dict, count: int) -> list[dict[str, int]]:
    """Promesses envoyées, engagements formés, tenus et rompus, par faction."""
    counts = [
        {"pledges": 0, "commitments": 0,
         "commitments_honoured": 0, "commitments_broken": 0}
        for _ in range(count)
    ]

    def bump(faction_id: int, field_name: str) -> None:
        if 0 <= faction_id < count:
            counts[faction_id][field_name] += 1

    for report in result.get("history", []):
        for event in report.get("events", []):
            kind = event.get("kind")
            if kind == "negotiation_delivered":
                bump(int(event["faction"]), "pledges")
            elif kind == "commitment_formed":
                for party in event.get("parties", []):
                    bump(int(party), "commitments")
            elif kind == "commitment_honoured":
                for party in event.get("parties", []):
                    bump(int(party), "commitments_honoured")
            elif kind == "commitment_betrayed":
                bump(int(event["faction"]), "commitments_broken")
    return counts


def _absorb_policy_reports(campaign: Campaign, result: dict, seats: list[str]) -> None:
    """Verse les compteurs propres aux policies LLM dans l'exactitude.

    Sans eux, un modèle **muet** — qui ne renvoie rien d'exploitable — afficherait
    une exactitude parfaite, puisqu'il n'a jamais proposé de coup illégal.
    """
    from controla.env.controla_env import faction_id_of

    for agent, report in result.get("policy_reports", {}).items():
        faction_id = faction_id_of(agent)
        if not 0 <= faction_id < len(seats):
            continue
        counters = campaign._counters(seats[faction_id])
        policy = report.get("policy", {})
        backend = report.get("backend", {})
        for key in ("planned_moves", "rejected_moves", "empty_plans",
                    "mute_turns", "truncated_turns", "filtered_turns",
                    "blank_answers", "blank_turns", "retries"):
            counters.llm[key] = counters.llm.get(key, 0) + int(policy.get(key, 0))
        counters.llm["calls"] = counters.llm.get("calls", 0) + int(
            backend.get("calls", 0)
        )
        counters.llm["failures"] = counters.llm.get("failures", 0) + sum(
            backend.get("failures", {}).values()
        )
        # Le budget et le cache, siège par siège : c'est ce que l'étude mémoire
        # compare. Ces compteurs sont PAR PARTIE (cf. `LLMPolicy.call_stats`),
        # donc leur somme sur la campagne est juste.
        for key in TOKEN_KEYS:
            counters.llm[key] = counters.llm.get(key, 0) + int(backend.get(key, 0) or 0)
        counters.llm["memory_seats"] = counters.llm.get("memory_seats", 0) + int(
            bool(report.get("memory"))
        )
