"""La boucle de calibration (spec §13) : faire varier les constantes, mesurer.

§13 interdit de calibrer à l'intuition. Ce module est l'outil qui rend cette
interdiction tenable : on décrit un plan de variantes en YAML, chaque variante joue
la même campagne sur les mêmes seeds et avec les mêmes rotations, et il en ressort
un tableau où seule la constante change.

**Zéro appel LLM.** Calibrer ne doit rien coûter — le panel est scripté, et il
tourne à une trentaine de secondes pour 48 parties.

Deux modes, et le premier est le bon par défaut :

* `axis` — un levier à la fois, chacun comparé à la référence. C'est l'expérience
  qui **attribue** un effet à une cause. Le coût croît en somme des valeurs.
* `grid` — produit cartésien, pour les rares couples de leviers dont on sait déjà
  qu'ils interagissent (mesuré : `attrition_loser` et `war_threshold` tirent en sens
  opposés). Le coût croît en produit, donc à réserver à deux ou trois axes.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from controla.arena.tournament import TournamentSpec, run_tournament
from controla.config.loader import explode_dotted, load_config
from controla.arena.storage import write_manifest

BASELINE = "référence"


@dataclass
class Cell:
    """Une variante : un jeu d'overrides et l'étiquette qui les résume."""

    label: str
    overrides: dict[str, object] = field(default_factory=dict)

    def config(self, base_path: str | None = None):
        return load_config(base_path, overrides=explode_dotted(self.overrides))


@dataclass
class SweepSpec:
    """Le plan de calibration, tel qu'écrit dans le YAML."""

    name: str
    panels: dict[str, list[str]]
    seeds: list[int]
    axes: dict[str, list] = field(default_factory=dict)
    mode: str = "axis"
    rotations: int = 0
    baseline: dict = field(default_factory=dict)
    config_path: str | None = None
    run_dir: Path | None = None

    def cells(self) -> list[Cell]:
        """Les variantes à jouer, référence en tête, sans doublon.

        Le dédoublonnage porte sur la configuration **résolue**, pas sur le dict
        d'overrides : un axe balayé autour de sa valeur courante contient forcément
        cette valeur, et cette variante-là est la référence. Sans quoi un plan de
        six axes rejouerait six fois la même campagne.
        """
        cells = [Cell(BASELINE, dict(self.baseline))]
        if self.mode == "grid":
            names = list(self.axes)
            for combination in itertools.product(*(self.axes[key] for key in names)):
                overrides = dict(self.baseline)
                overrides.update(dict(zip(names, combination)))
                label = " · ".join(
                    f"{_short(key)}={value}" for key, value in zip(names, combination)
                )
                cells.append(Cell(label, overrides))
        else:
            for key, values in self.axes.items():
                for value in values:
                    overrides = dict(self.baseline)
                    overrides[key] = value
                    cells.append(Cell(f"{_short(key)}={value}", overrides))

        unique: list[Cell] = []
        seen: set[str] = set()
        for cell in cells:
            signature = json.dumps(
                cell.config(self.config_path).to_dict(), sort_keys=True
            )
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(cell)
        return unique

    def games(self) -> int:
        """Nombre total de parties du plan — à connaître AVANT de le lancer."""
        rotations = self.rotations or min(len(seats) for seats in self.panels.values())
        return len(self.cells()) * len(self.panels) * len(self.seeds) * rotations

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "panels": {key: list(value) for key, value in self.panels.items()},
            "seeds": list(self.seeds),
            "rotations": self.rotations,
            "axes": {key: list(value) for key, value in self.axes.items()},
            "baseline": dict(self.baseline),
            "config_path": self.config_path,
        }


def _short(dotted: str) -> str:
    """`combat.attrition_loser` → `attrition_loser`, pour des étiquettes lisibles."""
    return str(dotted).rsplit(".", 1)[-1]


def load_sweep(path, run_dir=None) -> SweepSpec:
    """Lit un plan de calibration. Les seeds acceptent `{from: 1, to: 12}`."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    panels = data.get("panels") or {}
    if not panels:
        raise ValueError(f"{path} : aucun panel — un sweep sans joueurs ne mesure rien")
    return SweepSpec(
        name=str(data.get("name", Path(path).stem)),
        panels={key: list(value) for key, value in panels.items()},
        seeds=_seeds(data.get("seeds", [1, 2, 3])),
        axes={key: list(value) for key, value in (data.get("axes") or {}).items()},
        mode=str(data.get("mode", "axis")),
        rotations=int(data.get("rotations", 0)),
        baseline=dict(data.get("baseline") or {}),
        config_path=data.get("config_path"),
        run_dir=Path(run_dir) if run_dir else None,
    )


def _seeds(raw) -> list[int]:
    if isinstance(raw, dict):
        return list(range(int(raw["from"]), int(raw["to"]) + 1))
    return [int(value) for value in raw]


def run_sweep(spec: SweepSpec, on_cell=None) -> dict:
    """Joue toutes les variantes et renvoie le manifeste comparatif.

    `on_cell(index, total, label, panel, manifest)` est appelé après chaque
    tournoi : c'est par là que le CLI affiche sa progression.
    """
    directory = Path(spec.run_dir) if spec.run_dir else (
        Path("runs") / f"sweep-{spec.name}"
    )
    directory.mkdir(parents=True, exist_ok=True)

    cells = spec.cells()
    started = time.perf_counter()
    rows: list[dict] = []
    total = len(cells) * len(spec.panels)
    step = 0

    for cell in cells:
        cfg = cell.config(spec.config_path)
        for panel, seats in spec.panels.items():
            manifest = run_tournament(
                TournamentSpec(
                    seats=list(seats),
                    seeds=list(spec.seeds),
                    rotations=spec.rotations,
                    run_dir=directory / _slug(cell.label) / panel,
                    keep_games=False,
                ),
                config=cfg,
            )
            rows.append({
                "cell": cell.label,
                "overrides": dict(cell.overrides),
                "panel": panel,
                "games": manifest["games"],
                "criteria": manifest["criteria"],
                "leaderboard": manifest["leaderboard"],
            })
            step += 1
            if on_cell is not None:
                on_cell(step, total, cell.label, panel, manifest)

    result = {
        "sweep": spec.name,
        "spec": spec.to_dict(),
        "cells": len(cells),
        "games": sum(row["games"] for row in rows),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "rows": rows,
    }
    write_manifest(directory, result)
    return result


def _slug(label: str) -> str:
    keep = [
        character if character.isalnum() or character in "._-=" else "_"
        for character in label
    ]
    return "".join(keep).strip("_") or "cell"
