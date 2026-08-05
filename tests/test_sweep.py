"""La boucle de calibration (§13) : elle doit rendre un verdict, pas un avis.

Ce que ces tests protègent : un plan qui ne paie pas deux fois la même mesure, des
critères chiffrés plutôt qu'appréciés, et un sweep qui refuse une constante qui
n'existe pas plutôt que de l'ignorer en silence.
"""

from __future__ import annotations

import pytest
import yaml

from controla.arena.metrics import Criteria
from controla.arena.rating import Rating
from controla.arena.storage import read_manifest
from controla.arena.sweep import BASELINE, SweepSpec, load_sweep, run_sweep
from controla.config.loader import explode_dotted, load_config
from controla.replay.render import render_criteria, render_sweep
from tests.helpers import base_config

PANEL = ["cooperator", "conqueror", "expansionist", "turtle"]


def write_plan(path, **overrides) -> str:
    plan = {
        "name": "essai",
        "seeds": {"from": 1, "to": 1},
        "panels": {"militaire": PANEL},
        "axes": {"combat.attrition_loser": [0.20, 0.36]},
    }
    plan.update(overrides)
    path.write_text(yaml.safe_dump(plan, allow_unicode=True), encoding="utf-8")
    return str(path)


# ── le plan ─────────────────────────────────────────────────────────────────

def test_a_dotted_path_becomes_a_nested_override():
    assert explode_dotted({"combat.attrition_loser": 0.2}) == {
        "combat": {"attrition_loser": 0.2}
    }
    assert explode_dotted({"world.max_turns": 20})["world"]["max_turns"] == 20


def test_a_sweep_never_pays_twice_for_the_same_measurement():
    """Un axe balayé autour de sa valeur courante contient cette valeur, et cette
    variante-là EST la référence."""
    current = float(base_config().combat.attrition_loser)
    spec = SweepSpec(
        name="essai", panels={"p": PANEL}, seeds=[1],
        axes={"combat.attrition_loser": [0.20, current, 0.36]},
    )
    labels = [cell.label for cell in spec.cells()]
    assert labels[0] == BASELINE
    assert len(labels) == 3, f"la valeur courante a été rejouée : {labels}"


def test_one_lever_at_a_time_by_default():
    """`axis` isole la cause ; `grid` croise, et ne doit servir qu'aux leviers dont
    on sait déjà qu'ils interagissent."""
    axes = {"combat.attrition_loser": [0.20, 0.36],
            "emergence.war_threshold": [1.2, 2.0]}
    common = dict(name="e", panels={"p": PANEL}, seeds=[1], axes=axes)

    by_axis = SweepSpec(**common, mode="axis").cells()
    by_grid = SweepSpec(**common, mode="grid").cells()
    assert len(by_axis) == 1 + 2 + 2
    assert len(by_grid) == 1 + 2 * 2
    assert all(len(cell.overrides) <= 1 for cell in by_axis[1:])


def test_the_game_count_is_known_before_running(tmp_path):
    spec = load_sweep(write_plan(tmp_path / "plan.yaml",
                                 seeds={"from": 1, "to": 3}))
    # 3 variantes × 1 panel × 3 seeds × 4 rotations
    assert spec.games() == 3 * 1 * 3 * 4


def test_an_unknown_constant_is_rejected_not_ignored(tmp_path):
    """Une faute de frappe dans un plan de sweep ne doit jamais se dégrader en
    « valeur par défaut » — on croirait avoir mesuré un levier inexistant."""
    spec = load_sweep(write_plan(
        tmp_path / "plan.yaml", axes={"combat.attrition_loosers": [0.2]},
    ))
    with pytest.raises(KeyError, match="attrition_loosers"):
        spec.cells()


def test_a_plan_without_players_is_refused(tmp_path):
    path = tmp_path / "vide.yaml"
    path.write_text(yaml.safe_dump({"name": "vide", "seeds": [1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="panel"):
        load_sweep(path)


def test_the_shipped_plan_loads_and_is_sane():
    """Le plan livré dans `configs/` est un livrable : il doit se charger."""
    spec = load_sweep("configs/sweep_balance.yaml")
    assert spec.mode == "axis"
    assert spec.panels and spec.axes
    assert spec.games() > 0


# ── les critères §13 ────────────────────────────────────────────────────────

def synthetic(wins_for: int, turns: int = 15, alive: bool = True) -> dict:
    seats = ["a", "b", "c", "d"]
    order = [wins_for] + [i for i in range(4) if i != wins_for]
    return {
        "seed": 1, "seats": seats, "turns_played": turns,
        "ranking": [
            {"rank": rank, "faction_id": fid, "alive": alive,
             "eliminated_turn": None if alive else 9}
            for rank, fid in enumerate(order)
        ],
    }


def test_the_three_criteria_are_numbers_not_opinions():
    cfg = base_config()
    criteria, rating = Criteria(), Rating()
    for index in range(8):
        result = synthetic(index % 4)
        criteria.observe_match(result, int(cfg.world.max_turns))
        rating.observe_match(result)

    report = criteria.report(cfg, rating.standings())
    assert report["horizon"]["reached_max"] == 1.0
    assert report["balance"]["domination"] == 0.25    # parfaitement partagé
    assert report["balance"]["spread"] == 1.0
    assert report["verdict"]["a_horizon"] is True
    assert report["verdict"]["b_balance"] is True
    # Sans `random_legal` au panel, le critère (c) ne s'évalue pas — et il ne doit
    # surtout pas être compté comme un échec.
    assert report["verdict"]["c_skill"] is None


def test_domination_is_caught_when_one_strategy_takes_everything():
    cfg = base_config()
    criteria, rating = Criteria(), Rating()
    for _ in range(8):
        result = synthetic(0)
        criteria.observe_match(result, int(cfg.world.max_turns))
        rating.observe_match(result)

    report = criteria.report(cfg, rating.standings())
    assert report["balance"]["domination"] == 1.0
    assert report["balance"]["spread"] == 0.0
    assert report["verdict"]["b_balance"] is False


def test_the_spread_separates_two_arenas_that_domination_confuses():
    """40/38/19/4 et 40/20/20/20 ont la même domination et ne sont pas la même
    arène : dans l'une, deux stratégies sont mortes."""
    from controla.arena.metrics import _normalised_entropy

    assert _normalised_entropy([40, 20, 20, 20]) > _normalised_entropy([40, 38, 19, 4])


def test_a_short_game_fails_the_horizon_criterion():
    cfg = base_config()
    criteria, rating = Criteria(), Rating()
    for _ in range(4):
        result = synthetic(0, turns=6, alive=False)
        criteria.observe_match(result, int(cfg.world.max_turns))
        rating.observe_match(result)

    report = criteria.report(cfg, rating.standings())
    assert report["verdict"]["a_horizon"] is False
    assert report["horizon"]["eliminations_per_game"] == 4.0


def test_the_criteria_render_without_a_chance_seat():
    cfg = base_config()
    criteria, rating = Criteria(), Rating()
    criteria.observe_match(synthetic(0), int(cfg.world.max_turns))
    rating.observe_match(synthetic(0))
    rendered = render_criteria(criteria.report(cfg, rating.standings()))
    assert "§13 (c)" in rendered and "—" in rendered


# ── bout en bout ────────────────────────────────────────────────────────────

def test_a_sweep_runs_and_compares(tmp_path):
    spec = load_sweep(
        write_plan(tmp_path / "plan.yaml", axes={"emergence.war_threshold": [2.5]}),
        run_dir=tmp_path / "run",
    )
    manifest = run_sweep(spec)

    assert manifest["cells"] == 2
    assert len(manifest["rows"]) == 2
    assert read_manifest(tmp_path / "run") == manifest
    for row in manifest["rows"]:
        assert set(row["criteria"]) == {"horizon", "balance", "skill", "verdict"}
    assert "domin." in render_sweep(manifest)


def test_a_sweep_keeps_no_replay_files(tmp_path):
    """Un JSONL pèse ~125 ko : un sweep qui les garderait écrirait des centaines de
    mégaoctets pour n'en lire qu'un tableau de nombres."""
    spec = load_sweep(
        write_plan(tmp_path / "plan.yaml", axes={"emergence.war_threshold": [2.5]}),
        run_dir=tmp_path / "run",
    )
    run_sweep(spec)
    assert not list((tmp_path / "run").rglob("game_*.jsonl"))


def test_a_variant_actually_changes_the_outcome(tmp_path):
    """Sans cela, le sweep pourrait tourner sans que l'override atteigne le moteur."""
    spec = load_sweep(
        write_plan(tmp_path / "plan.yaml",
                   axes={"combat.building_capture_chance": [0.0, 1.0]}),
        run_dir=tmp_path / "run",
    )
    rows = {row["cell"]: row for row in run_sweep(spec)["rows"]}
    conquered = {
        label: next(e["mean_rank"] for e in row["leaderboard"]
                    if e["name"] == "conqueror")
        for label, row in rows.items()
    }
    assert conquered["building_capture_chance=1.0"] < \
        conquered["building_capture_chance=0.0"], (
        f"l'override n'a pas atteint le moteur : {conquered}"
    )


def test_the_sweep_config_thresholds_exist():
    thresholds = load_config().calibration
    assert 0.0 < float(thresholds.max_domination) <= 1.0
    assert 0.0 < float(thresholds.min_horizon) <= 1.0
    assert float(thresholds.min_skill_margin) > 0.0
