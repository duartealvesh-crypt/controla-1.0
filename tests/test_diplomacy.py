"""Négociation, engagements et réputation — LA métrique du benchmark (§7).

Ces tests protègent la propriété qui donne son sens au projet : le compteur de
trahisons doit mesurer **la parole**, jamais le dé ni la position dans la ronde.
"""

from __future__ import annotations

import itertools

import pytest

from controla.diplomacy.commitment import (
    ECON_ALLIANCE,
    NO_COMMITMENT,
    WAR_ALLIANCE,
    Commitment,
    Pledge,
    active_commitments,
    register_pledge,
    verify,
)
from controla.registry.catalog import build_catalog
from controla.rng import KeyedRNG
from controla.state.game_state import create_state
from tests.helpers import base_config


@pytest.fixture
def world():
    cfg = base_config()
    catalog = build_catalog(cfg)
    state = create_state(cfg, catalog, KeyedRNG(1))
    state.turn = 3
    return cfg, catalog, state


def spend(state, faction_id: int, action_key: str, target) -> None:
    """Simule une action PAYÉE : c'est cela, et cela seul, qui engage un dieu."""
    state.faction(faction_id).spent_this_turn.append((action_key, target))


def pledge_both(state, cfg, first: int, second: int, kind: str, victim: int = -1):
    register_pledge(state, Pledge(first, second, kind, state.turn, victim), cfg)
    return register_pledge(state, Pledge(second, first, kind, state.turn, victim), cfg)


# ── formation ───────────────────────────────────────────────────────────────

def test_a_one_sided_pledge_binds_nobody(world):
    """Sinon un dieu pourrait engager son adversaire à son insu et ruiner sa
    réputation à sa place."""
    cfg, _catalog, state = world
    assert register_pledge(state, Pledge(0, 1, WAR_ALLIANCE, 3, victim=2), cfg) is None
    assert state.commitments == []


def test_mutual_pledges_form_a_commitment(world):
    cfg, _catalog, state = world
    commitment = pledge_both(state, cfg, 0, 1, WAR_ALLIANCE, victim=2)
    assert commitment is not None
    assert commitment.parties == (0, 1)
    assert commitment.victim == 2
    assert commitment.active_at(state.turn)


def test_pledges_must_name_the_same_victim(world):
    """« La MÊME cible » (§7) : promettre d'attaquer deux ennemis différents n'est
    pas un accord."""
    cfg, _catalog, state = world
    register_pledge(state, Pledge(0, 1, WAR_ALLIANCE, 3, victim=2), cfg)
    assert register_pledge(state, Pledge(1, 0, WAR_ALLIANCE, 3, victim=3), cfg) is None
    assert state.commitments == []


def test_an_empty_pledge_forms_nothing(world):
    cfg, _catalog, state = world
    assert pledge_both(state, cfg, 0, 1, NO_COMMITMENT) is None


def test_the_same_pair_is_not_bound_twice(world):
    cfg, _catalog, state = world
    pledge_both(state, cfg, 0, 1, ECON_ALLIANCE)
    pledge_both(state, cfg, 0, 1, ECON_ALLIANCE)
    assert len(active_commitments(state, kind=ECON_ALLIANCE)) == 1


# ── vérification : l'intention engagée, pas l'issue ─────────────────────────

def test_a_war_alliance_is_honoured_when_both_pay_the_promised_call(world):
    cfg, catalog, state = world
    pledge_both(state, cfg, 0, 1, WAR_ALLIANCE, victim=2)
    spend(state, 0, "call_to_war", 2)
    spend(state, 1, "call_to_war", 2)

    events = verify(cfg, catalog, state)
    assert [event["kind"] for event in events] == ["commitment_honoured"]
    assert state.faction(0).reputation.honored == 1
    assert state.faction(1).reputation.honored == 1
    assert state.faction(0).reputation.betrayed == 0


def test_losing_the_battle_is_not_betraying(world):
    """LE point de la correction : on juge l'action payée, pas son résultat.

    Les deux alliés paient leur appel ; que leur peuple parte ou non, et qu'il
    gagne ou non, ne doit rien changer à la réputation. Sinon le compteur mesure
    le dé (§6 interdit au dieu d'ordonner une attaque).
    """
    cfg, catalog, state = world
    pledge_both(state, cfg, 0, 1, WAR_ALLIANCE, victim=2)
    spend(state, 0, "call_to_war", 2)
    spend(state, 1, "call_to_war", 2)

    # Aucune bataille n'a lieu, aucune envie ne franchit son seuil : sans importance.
    state.faction(0).war_desire[2] = 0.0
    state.faction(1).war_desire[2] = 0.0

    events = verify(cfg, catalog, state)
    assert events[0]["kind"] == "commitment_honoured"


def test_the_silent_ally_is_the_one_who_broke_the_promise(world):
    cfg, catalog, state = world
    pledge_both(state, cfg, 0, 1, WAR_ALLIANCE, victim=2)
    spend(state, 0, "call_to_war", 2)          # seul 0 tient parole

    events = verify(cfg, catalog, state)
    assert events[0]["kind"] == "commitment_betrayed"
    assert events[0]["faction"] == 1
    assert state.faction(1).reputation.betrayed == 1
    assert state.faction(0).reputation.betrayed == 0


def test_waiting_together_is_neither_honouring_nor_betraying(world):
    """L'obligation ne se déclenche que le tour où l'un passe à l'acte."""
    cfg, catalog, state = world
    pledge_both(state, cfg, 0, 1, WAR_ALLIANCE, victim=2)
    assert verify(cfg, catalog, state) == []
    assert state.faction(0).reputation.total == 0


def test_calling_war_on_your_own_ally_is_the_clearest_betrayal(world):
    cfg, catalog, state = world
    pledge_both(state, cfg, 0, 1, WAR_ALLIANCE, victim=2)
    spend(state, 0, "call_to_war", 1)          # 0 se retourne contre son allié

    events = verify(cfg, catalog, state)
    assert events[0]["kind"] == "commitment_betrayed"
    assert events[0]["faction"] == 0


def test_an_embargo_breaks_an_economic_alliance(world):
    cfg, catalog, state = world
    pledge_both(state, cfg, 0, 2, ECON_ALLIANCE)
    spend(state, 2, "trade_embargo", 0)

    events = verify(cfg, catalog, state)
    assert events[0]["kind"] == "commitment_betrayed"
    assert events[0]["faction"] == 2


def test_an_untouched_economic_alliance_is_honoured_each_turn(world):
    cfg, catalog, state = world
    pledge_both(state, cfg, 0, 2, ECON_ALLIANCE)
    events = verify(cfg, catalog, state)
    assert events[0]["kind"] == "commitment_honoured"


def test_a_broken_commitment_stops_being_verified(world):
    """Une trahison compte UNE fois : sinon un seul manquement ruinerait un dieu
    sur tous les tours restants."""
    cfg, catalog, state = world
    pledge_both(state, cfg, 0, 1, WAR_ALLIANCE, victim=2)
    spend(state, 0, "call_to_war", 2)
    verify(cfg, catalog, state)
    assert state.faction(1).reputation.betrayed == 1

    state.faction(0).spent_this_turn.clear()
    state.turn += 1
    spend(state, 0, "call_to_war", 2)
    verify(cfg, catalog, state)
    assert state.faction(1).reputation.betrayed == 1, "la trahison a été comptée deux fois"


def test_an_expired_commitment_is_no_longer_binding(world):
    cfg, catalog, state = world
    commitment = pledge_both(state, cfg, 0, 1, WAR_ALLIANCE, victim=2)
    state.turn = commitment.expires_turn + 1
    spend(state, 0, "call_to_war", 2)
    assert verify(cfg, catalog, state) == []


def test_verification_ignores_actions_by_name_and_reads_effects(world):
    """La vérification repère l'action par son EFFET, pas par son nom : un mod qui
    ajoute une action belliqueuse est couvert sans recâblage."""
    cfg, catalog, state = world
    war_actions = [
        spec.key for spec in catalog.actions
        if any(effect.kind == "war_desire" for effect in spec.effects)
    ]
    assert war_actions, "aucune action porteuse de war_desire dans le catalogue"

    pledge_both(state, cfg, 0, 1, WAR_ALLIANCE, victim=3)
    for faction_id in (0, 1):
        spend(state, faction_id, war_actions[0], 3)
    assert verify(cfg, catalog, state)[0]["kind"] == "commitment_honoured"


def test_reputation_is_public_in_every_observation():
    """Le coût de la trahison est SOCIAL : il doit être vu de tous (§7)."""
    from controla.env.controla_env import ControlaEnv

    env = ControlaEnv()
    env.reset(seed=2)
    env.state_.faction(1).reputation.betrayed = 4
    for agent in env.agents:
        env._refresh_info(agent)
        rivals = env.infos[agent]["view"]["rivals"]
        seen = {rival["id"]: rival["reputation"]["betrayed"] for rival in rivals}
        assert seen.get(1, 4) == 4 or agent.endswith("1")


def test_economic_alliance_forces_the_trade_relation_above_the_threshold(world):
    """« Transfert réciproque garanti » (§7) : l'alliance ouvre le canal, et le
    commerce automatique fait le reste — pas de mécanique parallèle."""
    from controla.diplomacy.commitment import enforce_economic_transfers

    cfg, _catalog, state = world
    pledge_both(state, cfg, 0, 3, ECON_ALLIANCE)
    enforce_economic_transfers(cfg, state)
    assert state.faction(0).trade_relation[3] >= float(cfg.trade.threshold)
    assert state.faction(3).trade_relation[0] >= float(cfg.trade.threshold)


# ── bout en bout, sur le panel scripté ──────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1])
def test_commitments_actually_form_in_real_games(seed):
    """Sans réciprocité de la part du panel, la métrique resterait vide."""
    from tests.helpers import run_scripted_match

    result = run_scripted_match(
        seed, seats=("cooperator", "betrayer", "tit_for_tat", "coalition_builder")
    )
    formed = [
        event for report in result["history"] for event in report["events"]
        if event.get("kind") == "commitment_formed"
    ]
    assert formed, "aucun engagement formé sur une partie entière"


def test_the_betrayal_metric_discriminates_the_archetypes():
    """Le critère de sortie de la tranche 3.

    Un coopérateur qui ne s'engage que sur ce qu'il peut tenir ne trahit jamais ;
    un traître, si. Sans cet écart, le benchmark ne mesurerait rien de la théorie
    de l'esprit.
    """
    from tests.helpers import run_scripted_match

    seats = ["cooperator", "betrayer", "tit_for_tat", "coalition_builder"]
    tally = {name: {"honored": 0, "betrayed": 0} for name in seats}
    for seed in range(6):
        for permutation in list(itertools.permutations(range(4)))[:2]:
            line = [seats[index] for index in permutation]
            result = run_scripted_match(seed, seats=line)
            for entry in result["ranking"]:
                name = line[entry["faction_id"]]
                tally[name]["honored"] += entry["reputation"]["honored"]
                tally[name]["betrayed"] += entry["reputation"]["betrayed"]

    assert tally["cooperator"]["betrayed"] == 0, (
        "un coopérateur ne s'engage que sur ce qu'il tiendra : "
        f"{tally['cooperator']}"
    )
    assert tally["betrayer"]["betrayed"] > 0, f"le traître n'a jamais trahi : {tally['betrayer']}"

    def rate(name: str) -> float:
        counts = tally[name]
        total = counts["honored"] + counts["betrayed"]
        return counts["honored"] / total if total else 0.0

    assert rate("cooperator") > rate("betrayer"), (
        f"la parole ne discrimine pas : {tally}"
    )


def test_a_commitment_survives_serialisation():
    """Le replay doit pouvoir relire les engagements d'une partie."""
    commitment = Commitment(
        kind=WAR_ALLIANCE, parties=(0, 2), victim=1,
        formed_turn=2, expires_turn=6,
    )
    payload = commitment.to_dict()
    assert payload["parties"] == [0, 2]
    assert payload["victim"] == 1
    assert payload["breached_turn"] is None
