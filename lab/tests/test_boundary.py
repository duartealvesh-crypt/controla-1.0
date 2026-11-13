"""La frontière entre le moteur et le laboratoire.

Le laboratoire ne voit du jeu que `controla.api` ; le jeu ne voit jamais le
laboratoire. Sans ce test, la frontière s'érode au premier raccourci.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {"controla.api"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            # `from controla import api` → controla.api
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def _top(name: str) -> str:
    return name.split(".")[0]


def test_lab_only_uses_the_engine_api():
    offenders = {}
    for path in (ROOT / "lab").rglob("*.py"):
        engine = {name for name in _imports(path) if _top(name) == "controla"}
        # `from controla import api` produit "controla" et "controla.api".
        engine.discard("controla")
        bad = engine - ALLOWED
        if bad:
            offenders[str(path.relative_to(ROOT))] = sorted(bad)
    assert not offenders, f"le laboratoire importe le moteur hors de controla.api : {offenders}"


def test_engine_never_imports_the_lab():
    offenders = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "controla").rglob("*.py")
        if any(_top(name) == "lab" for name in _imports(path))
    ]
    assert not offenders, f"le moteur importe le laboratoire : {offenders}"
