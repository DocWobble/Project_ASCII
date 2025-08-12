from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from .grid import Grid

def connectedness_energy(g: Grid, foreground: List[str]) -> float:
    """Lower is better: 0 if there is a single connected component of non-background cells."""
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
    return max(0.0, float(comps - 1))


def symmetry_energy(g: Grid, foreground: List[str]) -> float:
    """Horizontal bilateral symmetry cost (0 is perfectly symmetric)."""
    mask = np.isin(g.data, foreground).astype(int)
    flipped = np.fliplr(mask)
    return float(np.sum(np.abs(mask - flipped))) / float(g.h * g.w)


def density_energy(g: Grid, foreground: List[str], target_fill: float = 0.35) -> float:
    """Penalty for deviating from a target foreground density."""
    mask = np.isin(g.data, foreground)
    fill = float(mask.mean())
    return abs(fill - target_fill)


def anchor_energy(g: Grid, anchors: List[str]) -> float:
    """Penalty if any expected anchor glyphs are absent."""
    penalty = 0.0
    for a in anchors:
        if not (g.data == a).any():
            penalty += 0.5
    return penalty


def listener_energy(prompt: str, g: Grid, listener: Optional[object]) -> float:
    """
    Negative of the listener score so that higher semantic alignment lowers total energy.
    If no listener is provided or it errors, return 0.
    """
    if listener is None:
        return 0.0
    try:
        score = listener.score(prompt, g)
    except Exception:
        score = 0.0
    return -float(score)


def total_energy(
    g: Grid,
    anchors: List[str],
    foreground: List[str],
    prompt: str = "",
    listener: Optional[object] = None,
    listener_weight: float = 1.0,
  Dict[str, float]:

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from .grid import Grid


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
    return max(0.0, float(comps - 1))


def symmetry_energy(g: Grid, foreground: List[str]) -> float:
    """Horizontal bilateral symmetry cost."""
    mask = np.isin(g.data, foreground).astype(int)
    flipped = np.fliplr(mask)
    return float(np.sum(np.abs(mask - flipped))) / (g.h * g.w)


def density_energy(g: Grid, foreground: List[str], target_fill: float = 0.35) -> float:
    mask = np.isin(g.data, foreground)
    fill = float(mask.mean())
    return abs(fill - target_fill)


def anchor_energy(g: Grid, anchors: List[str]) -> float:
    """Penalty if anchors absent."""
    penalty = 0.0
    for a in anchors:
        if not (g.data == a).any():
            penalty += 0.5
    return penalty


def listener_energy(prompt: str, g: Grid, listener: Optional[object]) -> float:
    """
    Negative of listener score so that higher semantic alignment lowers total energy.
    If no listener is provided or it errors, return 0.
    """
    if listener is None:
        return 0.0
    try:
        score = listener.score(prompt, g)
    except Exception:
        score = 0.0
    return -float(score)


def total_energy(*args, **kwargs) -> Dict[str, float]:
    """
    Accepts BOTH signatures for backward/forward compatibility:

      A) total_energy(g, anchors, foreground)
      B) total_energy(prompt, g, anchors, foreground, listener=None, listener_weight=1.0)

    Keyword-only extras also supported: prompt=..., listener=..., listener_weight=...
    """
    # Detect which signature we got.
    if len(args) >= 3 and isinstance(args[0], Grid):
        # Signature A or A + keyword extras
        g: Grid = args[0]
        anchors: List[str] = args[1]
        foreground: List[str] = args[2]
        prompt: str = kwargs.get("prompt", "")
        listener = kwargs.get("listener", None)
        listener_weight: float = float(kwargs.get("listener_weight", 1.0))
    else:
        # Signature B (prompt first) or fully keyword
        prompt: str = args[0] if len(args) > 0 else kwargs.get("prompt", "")
        g: Grid = args[1] if len(args) > 1 else kwargs["g"]
        anchors: List[str] = args[2] if len(args) > 2 else kwargs.get("anchors", [])
        foreground: List[str] = args[3] if len(args) > 3 else kwargs.get("foreground", [])
        listener = kwargs.get("listener", None)
        listener_weight: float = float(kwargs.get("listener_weight", 1.0))

    terms: Dict[str, float] = {
        "connected": connectedness_energy(g, foreground),
        "symmetry": symmetry_energy(g, foreground),
        "density": density_energy(g, foreground),
        "anchors": anchor_energy(g, anchors),
    }
    # Listener lowers energy when it likes the slate
    terms["listener"] = listener_weight * listener_energy(prompt, g, listener)
    terms["total"] = sum(terms.values())
    return terms

    }
    terms["total"] = sum(terms.values())
    return terms
