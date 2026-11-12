"""Sprites pixel art PROVISOIRES de la carte, dessinés en texte.

Chaque sprite est une grille 16×16 de caractères ; chaque caractère renvoie à
une couleur de sa palette, `.` = transparent. Les terrains couvrent toute la
case ; les bâtiments se posent par-dessus, sur fond transparent.

Régénérer après modification :

    python -m controla.ui.make_sprites

Aucune dépendance : le PNG est écrit à la main (zlib + struct). Les vrais
assets remplaceront ces fichiers sans toucher au code — seuls les noms
(`<clé du catalogue>.png`) comptent.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "static" / "sprites"
SIZE = 16

OUTLINE = "#20160f"

SPRITES: dict[str, tuple[dict[str, str], list[str]]] = {
    # ── Terrains ─────────────────────────────────────────────────────────
    "plain": (
        {"G": "#5a8a3c", "g": "#4a7a30", "l": "#78a84a", "y": "#d8d070"},
        [
            "GGGGGGGGGGGGGGGG",
            "GGlGGGGGGGgGGGGG",
            "GGGGGGGGGGGGGGlG",
            "GGGGGgGGGGGGGGGG",
            "GGGGGGGGGyGGGGGG",
            "GlGGGGGGGGGGGGGG",
            "GGGGGGGGGGGGgGGG",
            "GGGGGGlGGGGGGGGG",
            "GGgGGGGGGGGGGGGG",
            "GGGGGGGGGGGlGGGG",
            "GGGGGGGGGGGGGGGG",
            "GGGGyGGGGGGGGGgG",
            "GGGGGGGGgGGGGGGG",
            "GlGGGGGGGGGGGGGG",
            "GGGGGGGGGGGGlGGG",
            "GGGGGgGGGGGGGGGG",
        ],
    ),
    "forest": (
        {"D": "#2f5a2a", "T": "#1f4224", "t": "#3c7a34", "b": "#5a3a20"},
        [
            "DDDDDDDDDDDDDDDD",
            "DDDtDDDDDDDDDDDD",
            "DDtttDDDDDDDtDDD",
            "DDtTtDDDDDDtttDD",
            "DtttTtDDDDDttTtD",
            "DtTttTDDDDtttTtD",
            "DDDbDDDDDDtTtttD",
            "DDDbDDDtDDDDbDDD",
            "DDDDDDtttDDDbDDD",
            "DDDDDDtTtDDDDDDD",
            "DDDDDtttTtDDDDDD",
            "DDDDDtTttTDDDtDD",
            "DDDDDDDbDDDDtttD",
            "DtDDDDDbDDDDtTtD",
            "tttDDDDDDDDtttTt",
            "DbDDDDDDDDDDDbDD",
        ],
    ),
    "river": (
        {"W": "#3a6ea8", "w": "#5a92cc", "s": "#cfe6ff"},
        [
            "WWWWWWWWWWWWWWWW",
            "WWWWWWWWWWWWWWWW",
            "WWwwWWWWWWWWWWWW",
            "WwWWwWWWWWWwwWWW",
            "WWWWWWWWWWwWWwWW",
            "WWWWWWWWWWWWWWWW",
            "WWWWWsWWWWWWWWWW",
            "WWWWWWWWWWWWWWWW",
            "WWWWWWWWwwWWWWWW",
            "WWWWWWWwWWwWWWWW",
            "WWwwWWWWWWWWWWWW",
            "WwWWwWWWWWWWWsWW",
            "WWWWWWWWWWWWWWWW",
            "WWWWWWWWWWwwWWWW",
            "WWWWWWWWWwWWwWWW",
            "WWWWWWWWWWWWWWWW",
        ],
    ),
    # ── Bâtiments (fond transparent) ─────────────────────────────────────
    "city": (
        {"O": OUTLINE, "R": "#b84a3a", "r": "#8c3428",
         "H": "#e8dcc0", "h": "#bfae8c", "K": "#3a2a20"},
        [
            "................",
            "................",
            "....OO..........",
            "...ORRO.........",
            "..ORRRRO........",
            ".ORRRRRrO...OO..",
            "ORRRRRRRrO.ORRO.",
            "OOOOOOOOOOORRRrO",
            ".OHHHHHHhOOOOOOO",
            ".OHKKHHHhO.OHHhO",
            ".OHKKHHHhO.OHKhO",
            ".OHHHHKKhO.OHKhO",
            ".OHHHHKKhO.OHHhO",
            ".OOOOOOOOO.OOOOO",
            "................",
            "................",
        ],
    ),
    "lab": (
        {"O": OUTLINE, "S": "#c8ccd8", "s": "#8c90a0", "K": "#3a2a20",
         "B": "#5ad0e0", "b": "#2a8ca0"},
        [
            "................",
            ".......OO.......",
            "......OBBO......",
            ".....OBbbBO.....",
            ".....OBbbBO.....",
            "......OBBO......",
            ".....OOOOOO.....",
            "....OSSSSSsO....",
            "....OSSKSSsO....",
            "....OSSSSSsO....",
            "....OSKSSKsO....",
            "....OSSSSSsO....",
            "...OSSSSSSSsO...",
            "...OSSSKKSSsO...",
            "...OOOOOOOOOO...",
            "................",
        ],
    ),
    "camp": (
        {"O": OUTLINE, "N": "#c8a060", "n": "#8c6a3a", "K": "#3a2a20",
         "P": "#5a3a20", "F": "#d04040"},
        [
            "................",
            "................",
            "........PFFF....",
            "........PFFFF...",
            "........PFF.....",
            "........P.......",
            ".......OPO......",
            "......ONPnO.....",
            ".....ONNPnnO....",
            "....ONNNKnnnO...",
            "...ONNNKKKnnnO..",
            "..ONNNNKKKnnnnO.",
            ".ONNNNNKKKnnnnnO",
            ".OOOOOOOOOOOOOOO",
            "................",
            "................",
        ],
    ),
    "field": (
        {"U": "#7a5a30", "u": "#5a4020", "Y": "#e0c050", "y": "#b8962c"},
        [
            "................",
            ".uuuuuuuuuuuuuu.",
            ".YyYYyYYyYYyYYy.",
            ".YYyYYyYYyYYyYY.",
            ".uuuuuuuuuuuuuu.",
            ".UUUUUUUUUUUUUU.",
            ".YyYYyYYyYYyYYy.",
            ".YYyYYyYYyYYyYY.",
            ".uuuuuuuuuuuuuu.",
            ".UUUUUUUUUUUUUU.",
            ".YyYYyYYyYYyYYy.",
            ".YYyYYyYYyYYyYY.",
            ".uuuuuuuuuuuuuu.",
            ".UUUUUUUUUUUUUU.",
            ".YyYYyYYyYYyYYy.",
            "................",
        ],
    ),
}


def _rgba(hex_color: str) -> bytes:
    value = hex_color.lstrip("#")
    return bytes(int(value[i:i + 2], 16) for i in (0, 2, 4)) + b"\xff"


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def encode_png(palette: dict[str, str], rows: list[str]) -> bytes:
    """PNG RGBA 8 bits d'une grille de caractères."""
    if len(rows) != SIZE or any(len(row) != SIZE for row in rows):
        raise ValueError(f"grille attendue {SIZE}×{SIZE}")
    colors = {key: _rgba(value) for key, value in palette.items()}
    colors["."] = b"\x00\x00\x00\x00"
    raw = b"".join(b"\x00" + b"".join(colors[c] for c in row) for row in rows)
    header = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(raw, 9)) + _chunk(b"IEND", b""))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (palette, rows) in SPRITES.items():
        (OUT_DIR / f"{name}.png").write_bytes(encode_png(palette, rows))
        print(f"{name}.png")


if __name__ == "__main__":
    main()
