"""Découverte des parties et des campagnes sur le disque.

Le format de stockage est déjà défini (`arena/storage.py`) : un JSONL par
partie, un `manifest.json` par campagne. Ce module ne fait que les ÉNUMÉRER et
en extraire de quoi peupler une liste — il ne réinterprète rien.

Une seule règle non négociable : **tout chemin demandé par le navigateur est
résolu puis vérifié comme étant sous la racine**. Un serveur local reste un
serveur ; `../../.env` est un chemin parfaitement valide tant que personne ne
regarde.
"""

from __future__ import annotations

import json
from pathlib import Path

from controla.arena.storage import load_match, read_manifest

DEFAULT_ROOT = Path("runs")


class OutsideRoot(ValueError):
    """Chemin sortant de la racine servie."""


def resolve_within(root: Path, relative: str) -> Path:
    """Résout `relative` sous `root`, ou lève `OutsideRoot`.

    `resolve()` d'abord, comparaison ensuite : sans cela un `..` ou un lien
    symbolique passerait la vérification textuelle et échapperait quand même.
    """
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise OutsideRoot(f"chemin hors de la racine servie : {relative!r}")
    return candidate


def _leader(manifest: dict) -> dict | None:
    board = manifest.get("leaderboard") or []
    return board[0] if board else None


def list_runs(root: Path) -> list[dict]:
    """Les campagnes trouvées sous la racine, la plus récente d'abord."""
    root = Path(root)
    if not root.is_dir():
        return []

    runs = []
    for manifest_path in sorted(root.glob("**/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Un run interrompu en pleine écriture ne doit pas casser la liste :
            # c'est justement celui qu'on veut voir pour comprendre pourquoi.
            continue
        directory = manifest_path.parent
        leader = _leader(manifest)
        criteria = manifest.get("criteria", {})
        runs.append({
            "path": str(directory.relative_to(root)).replace("\\", "/"),
            "run": manifest.get("run", directory.name),
            "games": manifest.get("games", 0),
            "seats": (manifest.get("spec") or {}).get("seats", []),
            "wall_seconds": manifest.get("wall_seconds"),
            "leader": None if leader is None else leader.get("name"),
            "domination": (criteria.get("balance") or {}).get("domination"),
            "modified": manifest_path.stat().st_mtime,
            "files": len(manifest.get("files") or []),
        })
    runs.sort(key=lambda entry: entry["modified"], reverse=True)
    return runs


def list_games(root: Path, run_path: str) -> list[dict]:
    """Les parties d'une campagne, dans l'ordre où elles ont été jouées."""
    directory = resolve_within(Path(root), run_path)
    manifest = read_manifest(directory)
    names = manifest.get("files") or [
        path.name for path in sorted(directory.glob("game_*.jsonl"))
    ]
    root_resolved = Path(root).resolve()
    return [
        {
            "path": str((directory / name).relative_to(root_resolved)).replace("\\", "/"),
            "name": name,
        }
        for name in names
        if (directory / name).is_file()
    ]


def list_loose_games(root: Path) -> list[dict]:
    """Les parties isolées — celles d'un `play --out`, hors campagne."""
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.glob("**/*")):
        if path.suffix not in (".json", ".jsonl") or path.name == "manifest.json":
            continue
        if path.name.startswith("game_") and (path.parent / "manifest.json").is_file():
            continue                                  # déjà listée par sa campagne
        out.append({
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "name": path.name,
            "modified": path.stat().st_mtime,
        })
    out.sort(key=lambda entry: entry["modified"], reverse=True)
    return out


def load_game(root: Path, relative: str) -> dict:
    """Relit une partie, JSONL de campagne ou JSON de `play --out`."""
    return load_match(resolve_within(Path(root), relative))


def load_manifest(root: Path, run_path: str) -> dict:
    return read_manifest(resolve_within(Path(root), run_path))
