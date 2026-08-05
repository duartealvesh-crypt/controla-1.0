"""Stockage d'une campagne : **un JSONL par partie**, un manifeste par run.

Le JSONL est le format de replay parce qu'il est *streamable* : la ligne d'en-tête
identifie la partie, chaque tour est écrit dès qu'il est joué, la dernière ligne
porte le résultat. Une campagne LLM interrompue laisse donc derrière elle des
parties complètes et une partie lisible jusqu'à son dernier tour joué — là où un
gros JSON unique n'aurait rien laissé du tout.

La relecture reconstruit **exactement** le dictionnaire que `play_match` a renvoyé :
`replay/` n'a pas à savoir si la partie vient de la mémoire ou du disque.
"""

from __future__ import annotations

import json
from pathlib import Path

# Ce qui identifie la partie, et qui est donc écrit AVANT de la jouer.
HEADER_KEYS = ("seed", "seats")


def _line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def write_game(path, result: dict) -> Path:
    """Écrit une partie au format JSONL et renvoie son chemin."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = {key: result[key] for key in HEADER_KEYS if key in result}
    tail = {
        key: value for key, value in result.items()
        if key not in header and key != "history"
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write(_line({"record": "match", **header}) + "\n")
        for report in result.get("history", []):
            handle.write(_line({"record": "turn", **report}) + "\n")
        handle.write(_line({"record": "result", **tail}) + "\n")
    return path


def read_game(path) -> dict:
    """Relit un JSONL de partie. Une ligne tronquée est ignorée, pas fatale —
    c'est tout l'intérêt du format quand une campagne a été interrompue."""
    result: dict = {}
    history: list[dict] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if payload.pop("record", None) == "turn":
            history.append(payload)
        else:
            result.update(payload)
    result["history"] = history
    return result


def load_match(path) -> dict:
    """Relit une partie, JSONL (tournoi) ou JSON (`play --out`)."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return read_game(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(directory, payload: dict) -> Path:
    """Le manifeste : le classement, les métriques et de quoi refaire le run."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return path


def read_manifest(directory) -> dict:
    return json.loads(
        (Path(directory) / "manifest.json").read_text(encoding="utf-8")
    )


def game_files(directory) -> list[Path]:
    return sorted(Path(directory).glob("game_*.jsonl"))
