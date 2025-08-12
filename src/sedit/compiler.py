from __future__ import annotations
from typing import List
from .grid import Grid
from .palette import ANCHORS


def compile_to_text(prompt: str, grid: Grid) -> str:
    """Extremely simple verbalizer: look at anchors present."""
    text = "".join(grid.as_lines())
    chosen = None
    for k, anchors in ANCHORS.items():
        if any(a in text for a in anchors):
            chosen = k
            break
    if chosen:
        return f"A simple depiction of a {chosen}."
    return "A simple emoji mosaic."
