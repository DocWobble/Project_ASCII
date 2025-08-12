from __future__ import annotations
from typing import Dict, List
import numpy as np
from .grid import Grid
from .palette import BACKGROUND


def connectedness_energy(g: Grid, foreground: List[str]) -> float:
    """Lower is better: 0 if a single connected component of non-background exists."""
    mask = np.isin(g.data, foreground)
    visited = np.zeros_like(mask, dtype=bool)
    comps = 0
    for r in range(g.h):
        for c in range(g.w):
            if mask[r, c] and not visited[r, c]:
                comps += 1
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    for nr, nc in ((rr + 1, cc), (rr - 1, cc), (rr, cc + 1), (rr, cc - 1)):
                        if 0 <= nr < g.h and 0 <= nc < g.w and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
    return max(0, comps - 1)


def symmetry_energy(g: Grid, foreground: List[str]) -> float:
    """Horizontal bilateral symmetry cost."""
    mask = np.isin(g.data, foreground).astype(int)
    flipped = np.fliplr(mask)
    return float(np.sum(np.abs(mask - flipped))) / (g.h * g.w)


def density_energy(g: Grid, foreground: List[str], target_fill: float = 0.35) -> float:
    mask = np.isin(g.data, foreground)
    fill = mask.mean()
    return abs(fill - target_fill)


def anchor_energy(g: Grid, anchors: List[str]) -> float:
    """Penalty if anchors absent."""
    penalty = 0.0
    for a in anchors:
        present = (g.data == a).any()
        if not present:
            penalty += 0.5
    return penalty


def total_energy(g: Grid, anchors: List[str], foreground: List[str]) -> Dict[str, float]:
    terms = {
        "connected": connectedness_energy(g, foreground),
        "symmetry": symmetry_energy(g, foreground),
        "density": density_energy(g, foreground),
        "anchors": anchor_energy(g, anchors),
    }
    terms["total"] = sum(terms.values())
    return terms
