"""L'interface web : confinement des chemins, session jouable, audit de modèles.

Trois choses valent d'être verrouillées ici :

1. **le confinement.** Un serveur local reste un serveur, et `../../.env` est un
   chemin parfaitement valide tant que personne ne regarde ;
2. **la session jouable**, qui est la SEULE seconde boucle de jeu du projet et
   doit donc rester d'accord avec `play_match` ;
3. **l'audit de modèles**, dont le rôle est précisément de détecter des écarts —
   un audit qui ne mord pas est pire qu'aucun audit.
"""

from __future__ import annotations

import json

import pytest

from controla.arena.storage import write_game
from controla.policies.llm import catalogue as model_catalogue
from controla.registry.catalog import build_catalog
from controla.ui import library
from controla.ui.session import HUMAN_SEAT, Session, SessionSpec, SessionStore
from tests.helpers import base_config, run_scripted_match


# ── Confinement des chemins ─────────────────────────────────────────────────

@pytest.mark.parametrize("attempt", [
    "../.env",
    "../../secrets.txt",
    "sub/../../outside.json",
])
def test_paths_outside_the_root_are_refused(tmp_path, attempt):
    root = tmp_path / "runs"
    root.mkdir()
    with pytest.raises(library.OutsideRoot):
        library.resolve_within(root, attempt)


def test_paths_inside_the_root_are_accepted(tmp_path):
    root = tmp_path / "runs"
    (root / "campagne").mkdir(parents=True)
    resolved = library.resolve_within(root, "campagne/game_0000.jsonl")
    assert resolved.parent.name == "campagne"


# ── Bibliothèque ────────────────────────────────────────────────────────────

def test_library_lists_runs_and_loose_games(tmp_path):
    root = tmp_path / "runs"
    directory = root / "campagne"
    directory.mkdir(parents=True)

    result = run_scripted_match(seed=2, snapshots=True)
    write_game(directory / "game_0000.jsonl", result)
    (directory / "manifest.json").write_text(json.dumps({
        "run": "campagne", "games": 1, "files": ["game_0000.jsonl"],
        "spec": {"seats": list(result["seats"])},
        "leaderboard": [{"name": "turtle"}],
        "criteria": {"balance": {"domination": 0.5}},
    }), encoding="utf-8")
    (root / "solo.json").write_text(json.dumps(result), encoding="utf-8")

    runs = library.list_runs(root)
    assert [run["run"] for run in runs] == ["campagne"]
    assert runs[0]["leader"] == "turtle"

    games = library.list_games(root, "campagne")
    assert [game["name"] for game in games] == ["game_0000.jsonl"]

    # La partie de campagne n'est pas re-listée comme « isolée ».
    loose = library.list_loose_games(root)
    assert [entry["name"] for entry in loose] == ["solo.json"]

    reread = library.load_game(root, games[0]["path"])
    assert reread["seed"] == result["seed"]


def test_a_truncated_manifest_does_not_break_the_listing(tmp_path):
    """Une campagne interrompue est justement celle qu'on veut pouvoir ouvrir."""
    root = tmp_path / "runs"
    (root / "cassee").mkdir(parents=True)
    (root / "cassee" / "manifest.json").write_text("{ tronqué", encoding="utf-8")
    assert library.list_runs(root) == []


# ── Session jouable ─────────────────────────────────────────────────────────

def _session(seed=42, human_seat=0):
    cfg = base_config()
    catalog = build_catalog(cfg)
    seats = ["cooperator", "expansionist", "turtle", "conqueror"]
    seats[human_seat] = HUMAN_SEAT
    return Session.create(SessionSpec(seed=seed, seats=seats, human_seat=human_seat),
                          cfg, catalog)


def test_session_stops_and_waits_for_the_human():
    session = _session()
    session.advance()
    assert session.waiting_for_human
    assert not session.finished
    snapshot = session.snapshot()
    assert snapshot["view"]["faction_id"] == session.spec.human_seat
    assert snapshot["view"]["legal_actions"], "aucun coup légal proposé"


def test_a_human_move_costs_points_and_is_applied():
    session = _session()
    session.advance()
    view = session.snapshot()["view"]
    move = next(a for a in view["legal_actions"] if a["action"] != "end_turn")
    before = view["self"]["intervention_points"]

    session.submit(move["index"])
    after = session.snapshot()
    # Soit il reste des points et c'est encore à lui, soit le tour a basculé.
    if after["waiting_for_human"] and after["turn"] == 0:
        assert after["view"]["self"]["intervention_points"] == before - move["cost"]


def test_playing_out_a_session_produces_a_replayable_result():
    """Une partie jouée à la main se relit exactement comme une partie de banc."""
    session = _session()
    session.advance()

    guard = 0
    while not session.finished and guard < 500:
        guard += 1
        view = session.snapshot().get("view")
        if view is None:
            session.advance()
            continue
        end = next(a for a in view["legal_actions"] if a["action"] == "end_turn")
        session.submit(end["index"])

    assert session.finished, "la partie n'a pas terminé"
    result = session.result()
    # Les clés que `replay/render.py` et la visionneuse attendent.
    for key in ("seed", "seats", "turns_played", "ranking", "history", "world"):
        assert key in result, f"clé '{key}' absente du résultat"
    assert len(result["ranking"]) == 4
    assert all("board" in report for report in result["history"])
    from controla.replay.render import render_match
    assert render_match(result)                       # ne lève pas


def test_submitting_out_of_turn_is_refused():
    session = _session()
    session.advance()
    view = session.snapshot()["view"]
    end = next(a for a in view["legal_actions"] if a["action"] == "end_turn")
    session.submit(end["index"])
    # Après `end_turn`, la session a déjà ré-avancé ; on force le cas en jouant
    # une seconde fois sur une session qui n'attend plus.
    if not session.waiting_for_human:
        with pytest.raises(RuntimeError):
            session.submit(end["index"])


def test_session_store_evicts_the_oldest():
    store = SessionStore(limit=2)
    kept = []
    for _ in range(3):
        session = _session()
        store.add(session)
        kept.append(session.id)
    assert store.get(kept[0]) is None
    assert store.get(kept[-1]) is not None


# ── Audit de modèles ────────────────────────────────────────────────────────

CATALOGUE = {
    "vendor/bon": {
        "id": "vendor/bon",
        "context_length": 128000,
        "pricing": {"prompt": "0.0000001", "completion": "0.0000004"},
        "supported_parameters": ["tools", "temperature", "seed"],
    },
    "vendor/sans-temperature": {
        "id": "vendor/sans-temperature",
        "context_length": 400000,
        "pricing": {"prompt": "0.00000005", "completion": "0.0000004"},
        "supported_parameters": ["tools", "seed"],
    },
    "vendor/sans-tools": {
        "id": "vendor/sans-tools",
        "context_length": 8000,
        "pricing": {"prompt": "0.0000002", "completion": "0.0000002"},
        "supported_parameters": ["temperature"],
    },
}


def test_audit_accepts_a_coherent_declaration():
    report = model_catalogue.audit_model(
        "vendor/bon", {"provider": "v", "tool_support": True, "temperature": 0.7},
        CATALOGUE,
    )
    assert report["problems"] == []
    assert report["price_in"] == pytest.approx(0.1)


def test_audit_catches_an_unsupported_sampling_setting():
    """LE défaut que cet audit existe pour attraper : un réglage ignoré en
    silence fait comparer des conditions d'inférence différentes."""
    report = model_catalogue.audit_model(
        "vendor/sans-temperature",
        {"provider": "v", "tool_support": True, "temperature": 0.7},
        CATALOGUE,
    )
    assert any("temperature" in problem for problem in report["problems"])


def test_audit_is_silent_when_the_setting_is_explicitly_none():
    report = model_catalogue.audit_model(
        "vendor/sans-temperature",
        {"provider": "v", "tool_support": True, "temperature": None},
        CATALOGUE,
    )
    assert report["problems"] == []


def test_audit_catches_a_missing_model_and_missing_tool_support():
    absent = model_catalogue.audit_model(
        "vendor/fantome", {"provider": "v"}, CATALOGUE,
    )
    assert absent["known"] is False
    assert any("absent" in problem for problem in absent["problems"])

    no_tools = model_catalogue.audit_model(
        "vendor/sans-tools", {"provider": "v", "tool_support": True}, CATALOGUE,
    )
    assert any("tools" in problem for problem in no_tools["problems"])


def test_an_unreachable_catalogue_warns_rather_than_reassures():
    """Ne rien avoir pu vérifier n'est pas « tout va bien »."""
    report = model_catalogue.audit_model("vendor/bon", {"provider": "v"}, {})
    assert report["problems"] == []
    assert report["warnings"], "un audit aveugle doit le dire"


def test_campaign_cost_scales_with_the_number_of_games():
    report = model_catalogue.audit_model("vendor/bon", {"provider": "v"}, CATALOGUE)
    one = model_catalogue.campaign_cost(report, games=1)
    fifty = model_catalogue.campaign_cost(report, games=50)
    assert fifty == pytest.approx(one * 50)
    assert model_catalogue.campaign_cost({"price_in": None}, games=1) is None
