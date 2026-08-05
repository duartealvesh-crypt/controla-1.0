"""Campagne, classement OpenSkill et les cinq métriques §12 (tranche 4).

Ce que ces tests protègent : un classement **reproductible** (sinon le benchmark
mesure l'ordre d'exécution), des parties **relisibles** après coup (sinon il n'y a
pas de finding), et des métriques qui ne récompensent pas le silence.
"""

from __future__ import annotations

from controla.arena.metrics import Campaign, focus
from controla.arena.rating import Rating
from controla.arena.storage import (
    game_files,
    load_match,
    read_game,
    read_manifest,
    write_game,
)
from controla.arena.tournament import TournamentSpec, action_universe, run_tournament
from controla.registry.catalog import build_catalog
from controla.replay.render import render_leaderboard, render_match
from tests.helpers import base_config, run_scripted_match

PANEL = ["cooperator", "betrayer", "expansionist", "turtle"]


def synthetic_match(**overrides) -> dict:
    """Un résultat de partie minimal, pour tester les métriques sans jouer."""
    result = {
        "seed": 7,
        "seats": ["a", "b", "c", "d"],
        "turns_played": 4,
        "invalid_actions": {f"god_{index}": 0 for index in range(4)},
        "history": [],
        "ranking": [
            {
                "rank": index, "faction_id": index, "name": f"f{index}",
                "alive": True, "eliminated_turn": None, "score": 100.0 - index,
                "components": {"tiles": 20.0}, "reputation": {"honored": 0,
                                                              "betrayed": 0},
            }
            for index in range(4)
        ],
    }
    result.update(overrides)
    return result


def turn(index: int, *events: dict) -> dict:
    return {"turn": index, "events": list(events), "battles": [],
            "eliminations": [], "per_faction": {}}


def paid(faction_id: int, key: str) -> dict:
    return {"kind": "divine_action", "faction": faction_id, "action": key}


# ── stockage : le JSONL est le replay ───────────────────────────────────────

def test_a_stored_game_reads_back_identical(tmp_path):
    """Le replay doit voir exactement ce que l'arène a produit — sinon les
    métriques recalculées après coup porteraient sur autre chose.

    La référence est le résultat passé par JSON, pas l'objet en mémoire : une
    coordonnée de case est un tuple qui revient en liste, et c'est le format qui
    l'impose, pas le stockage.
    """
    import json

    result = run_scripted_match(3)
    write_game(tmp_path / "game_0000.jsonl", result)
    assert read_game(tmp_path / "game_0000.jsonl") == json.loads(
        json.dumps(result, ensure_ascii=False)
    )


def test_a_truncated_game_stays_readable(tmp_path):
    """L'intérêt du JSONL : une campagne interrompue laisse une partie lisible
    jusqu'à son dernier tour complet, au lieu d'un fichier illisible."""
    path = write_game(tmp_path / "game.jsonl", run_scripted_match(3))
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(lines[:4]) + "\n" + lines[4][:20], encoding="utf-8"
    )

    partial = read_game(path)
    assert partial["seed"] == 3
    assert len(partial["history"]) == 3


def test_load_match_accepts_both_formats(tmp_path):
    import json

    result = run_scripted_match(4)
    write_game(tmp_path / "game.jsonl", result)
    (tmp_path / "game.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )
    assert load_match(tmp_path / "game.jsonl") == load_match(tmp_path / "game.json")


def test_a_stored_game_renders(tmp_path):
    write_game(tmp_path / "game.jsonl", run_scripted_match(5))
    rendered = render_match(load_match(tmp_path / "game.jsonl"))
    assert "CLASSEMENT FINAL" in rendered


# ── classement : OpenSkill / Plackett-Luce ──────────────────────────────────

def test_the_ordinal_is_mu_minus_three_sigma():
    """C'est LA définition du classement prudent : on ne monte qu'une fois observé."""
    rating = Rating()
    rating.observe(["a", "b", "c", "d"], [0, 1, 2, 3])
    for standing in rating.standings():
        assert standing.ordinal == standing.mu - 3.0 * standing.sigma


def test_a_constant_winner_ends_on_top():
    rating = Rating()
    for _ in range(8):
        rating.observe(["a", "b", "c", "d"], [0, 1, 2, 3])
    table = [standing.name for standing in rating.standings()]
    assert table == ["a", "b", "c", "d"]


def test_the_ordinal_rewards_evidence_not_luck():
    """Même palmarès sur trois parties et sur quinze : la seconde vaut plus.
    Sans cela, un modèle chanceux sur trois parties coifferait un modèle prouvé."""
    def ordinal_after(games: int) -> float:
        rating = Rating()
        for _ in range(games):
            rating.observe(["a", "b", "c", "d"], [0, 1, 2, 3])
        return next(s for s in rating.standings() if s.name == "a").ordinal

    assert ordinal_after(15) > ordinal_after(3)


def test_a_participant_holding_two_seats_is_judged_on_both():
    """Ses deux sièges finissent premier et dernier : ils se compensent.

    Sans moyenne des postérieurs, le dernier siège traité écraserait l'autre et le
    participant serait jugé sur un seul de ses résultats — celui que l'ordre des
    sièges a désigné.
    """
    def mu_of(seats: list[str]) -> float:
        rating = Rating()
        rating.observe(seats, [0, 1, 2, 3])
        return next(s for s in rating.standings() if s.name == "base").mu

    won, lost = mu_of(["base", "a", "b", "c"]), mu_of(["a", "b", "c", "base"])
    both = mu_of(["base", "a", "b", "base"])
    assert lost < both < won, (
        f"un siège a écrasé l'autre : μ={both:.2f} hors de ]{lost:.2f}, {won:.2f}["
    )

    rating = Rating()
    rating.observe(["base", "a", "b", "base"], [0, 1, 2, 3])
    base = next(s for s in rating.standings() if s.name == "base")
    assert base.games == 1 and base.seats == 2


def test_two_seats_in_one_game_do_not_count_as_two_games():
    """La confiance doit rester adossée au nombre de PARTIES observées.

    OpenSkill resserre malgré tout un peu plus σ pour qui tient deux sièges — la
    partie contient réellement plus de comparaisons le concernant. La marge est
    mesurée ici pour qu'une régression vers un double comptage franc se voie.
    """
    doubled, independent = Rating(), Rating()
    for _ in range(6):
        doubled.observe(["a", "base", "base", "c"], [0, 1, 2, 3])
    for _ in range(12):
        independent.observe(["a", "base", "c", "d"], [0, 1, 2, 3])

    base = next(s for s in doubled.standings() if s.name == "base")
    reference = next(s for s in independent.standings() if s.name == "base")
    assert base.games == 6 and base.seats == 12
    assert base.sigma > reference.sigma * 0.9


def test_the_standings_break_ties_by_name():
    """Deux participants jamais départagés doivent sortir dans un ordre stable."""
    rating = Rating()
    rating.observe(["z", "a"], [0, 0])
    assert [standing.name for standing in rating.standings()] == ["a", "z"]


# ── métriques : les cinq familles ───────────────────────────────────────────

def test_every_participant_gets_the_five_metrics():
    campaign = Campaign()
    campaign.observe_match(synthetic_match())
    report = campaign.report(universe=9)
    assert set(report) == {"a", "b", "c", "d"}
    for entry in report.values():
        assert {"parole", "exactitude", "coherence", "alliances",
                "survie"} <= set(entry)


def test_focus_separates_a_single_line_from_scattering():
    assert focus({"bless": 10}, universe=9) == 1.0
    spread = {f"action_{index}": 5 for index in range(9)}
    assert focus(spread, universe=9) == 0.0
    assert 0.0 < focus({"bless": 8, "call_to_war": 2}, universe=9) < 1.0


def test_a_mute_god_is_not_credited_with_perfect_coherence():
    """LE piège de la métrique : un modèle muet ne propose aucun coup illégal.
    Sans `engagement` à côté, son silence passerait pour de la discipline."""
    campaign = Campaign()
    campaign.observe_match(synthetic_match(
        history=[turn(index) for index in range(1, 5)]
    ))
    coherence = campaign.report(universe=9)["a"]["coherence"]
    assert coherence["engagement"] == 0.0
    assert coherence["idle_turn_rate"] == 1.0
    assert coherence["focus"] is None


def test_dying_early_does_not_count_as_idleness():
    """Un dieu éliminé cesse de jouer : ses tours d'après ne sont pas de l'inaction."""
    ranking = synthetic_match()["ranking"]
    ranking[0] = {**ranking[0], "alive": False, "eliminated_turn": 2}
    campaign = Campaign()
    campaign.observe_match(synthetic_match(
        ranking=ranking,
        history=[turn(1, paid(0, "bless")), turn(2, paid(0, "bless")),
                 turn(3), turn(4)],
    ))
    coherence = campaign.report(universe=9)["a"]["coherence"]
    assert coherence["idle_turn_rate"] == 0.0
    assert coherence["engagement"] == 1.0


def test_accuracy_counts_rejected_moves_against_attempted_ones():
    campaign = Campaign()
    campaign.observe_match(synthetic_match(
        invalid_actions={"god_0": 3, "god_1": 0, "god_2": 0, "god_3": 0},
        history=[turn(1, paid(0, "bless"), paid(1, "bless"))],
    ))
    report = campaign.report(universe=9)
    assert report["a"]["exactitude"] == {"attempted": 4, "invalid": 3, "rate": 0.25}
    assert report["b"]["exactitude"]["rate"] == 1.0


def test_alliance_counters_follow_the_commitment_events():
    campaign = Campaign()
    campaign.observe_match(synthetic_match(history=[turn(
        1,
        {"kind": "negotiation_delivered", "faction": 0, "target": 1},
        {"kind": "commitment_formed", "parties": [0, 1]},
        {"kind": "commitment_honoured", "parties": [0, 1]},
        {"kind": "commitment_betrayed", "faction": 1, "loyal": 0,
         "parties": [0, 1]},
    )]))
    report = campaign.report(universe=9)
    assert report["a"]["alliances"] == {
        "pledges": 1, "formed": 1, "honoured_turns": 1, "broken": 0,
        "formed_per_game": 1.0,
    }
    assert report["b"]["alliances"]["broken"] == 1


def test_the_parole_metric_reads_the_public_reputation():
    ranking = synthetic_match()["ranking"]
    ranking[1] = {**ranking[1], "reputation": {"honored": 3, "betrayed": 1}}
    campaign = Campaign()
    campaign.observe_match(synthetic_match(ranking=ranking))
    assert campaign.report(universe=9)["b"]["parole"] == {
        "honoured": 3, "betrayed": 1, "rate": 0.75,
    }


def test_metrics_from_a_real_match_are_all_populated():
    """Les compteurs synthétiques ne prouvent pas le câblage sur une vraie partie."""
    campaign = Campaign()
    campaign.observe_match(run_scripted_match(11, seats=PANEL))
    report = campaign.report(action_universe(build_catalog(base_config())))
    for name in PANEL:
        entry = report[name]
        assert entry["exactitude"]["attempted"] > 0, name
        assert entry["coherence"]["focus"] is not None, name
        assert entry["survie"]["mean_rank"] is not None, name


# ── campagne ────────────────────────────────────────────────────────────────

def test_every_participant_plays_every_corner_exactly_once():
    """Les quatre coins sont symétriques, pas les voisinages : sans rotation, le
    classement mesurerait en partie la position."""
    spec = TournamentSpec(seats=PANEL, seeds=[1])
    positions = {name: set() for name in PANEL}
    for _rotation, line_up in spec.line_ups():
        for seat, name in enumerate(line_up):
            positions[name].add(seat)
    assert all(seats == {0, 1, 2, 3} for seats in positions.values())


def test_the_campaign_plan_is_deterministic_and_sized_as_announced():
    spec = TournamentSpec(seats=PANEL, seeds=[1, 2], rotations=2, repetitions=3)
    plan = spec.games()
    assert len(plan) == 2 * 2 * 3
    assert plan == TournamentSpec(
        seats=PANEL, seeds=[1, 2], rotations=2, repetitions=3
    ).games()


def test_a_campaign_writes_one_readable_jsonl_per_game(tmp_path):
    manifest = run_tournament(TournamentSpec(
        seats=PANEL, seeds=[1], rotations=2, run_dir=tmp_path,
    ))
    files = game_files(tmp_path)
    assert len(files) == manifest["games"] == 2
    assert read_manifest(tmp_path) == manifest
    for path in files:
        assert render_match(load_match(path)).count("\n") > 5


def test_a_campaign_ranks_and_measures_everyone(tmp_path):
    manifest = run_tournament(TournamentSpec(
        seats=PANEL, seeds=[1], rotations=2, run_dir=tmp_path,
    ))
    assert {entry["name"] for entry in manifest["leaderboard"]} == set(PANEL)
    assert set(manifest["metrics"]) == set(PANEL)
    ordinals = [entry["ordinal"] for entry in manifest["leaderboard"]]
    assert ordinals == sorted(ordinals, reverse=True)
    assert "μ−3σ" in render_leaderboard(manifest)


def test_the_same_campaign_replayed_gives_the_same_leaderboard(tmp_path):
    """Le critère de sortie de la tranche 4 : sans cela, le classement mesurerait
    l'ordre d'exécution autant que les participants."""
    def campaign(directory) -> list:
        return run_tournament(TournamentSpec(
            seats=PANEL, seeds=[2], rotations=2, run_dir=directory,
        ))["leaderboard"]

    assert campaign(tmp_path / "first") == campaign(tmp_path / "second")
