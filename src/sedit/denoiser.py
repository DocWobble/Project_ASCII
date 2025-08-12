from __future__ import annotations
import random
from typing import List, Callable, Dict
from dataclasses import dataclass
import numpy as np
from .grid import Grid
from .palette import PALETTE, BACKGROUND
from .energy import total_energy


@dataclass
class DiffusionConfig:
    steps: int = 12
    proposals_per_step: int = 64
    foreground: List[str] = None
    anchors: List[str] = None
    temperature: float = 1.0
    seed: int = 0


def propose_edits(g: Grid, palette: List[str], k: int) -> List[Grid]:
    outs = []
    h, w = g.h, g.w
    for _ in range(k):
        r, c = random.randrange(h), random.randrange(w)
        new_ch = random.choice(palette)
        if new_ch == g.get(r, c):
            continue
        gg = g.copy()
        gg.put(r, c, new_ch)
        outs.append(gg)
    return outs


def run_diffusion(prompt: str, g: Grid, cfg: DiffusionConfig, energy_fn: Callable[..., Dict[str, float]]):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.foreground is None:
        cfg.foreground = [p for p in PALETTE if p != BACKGROUND]
    if cfg.anchors is None:
        cfg.anchors = []
    history = [g.copy()]
    best = g.copy()
    best_e = energy_fn(best, cfg.anchors, cfg.foreground)["total"]
    for t in range(cfg.steps):
        candidates = propose_edits(best, PALETTE, cfg.proposals_per_step)
        scored = []
        for cand in candidates:
            terms = energy_fn(cand, cfg.anchors, cfg.foreground)
            scored.append((terms["total"], cand))
        if not scored:
            break
        scored.sort(key=lambda x: x[0])
        new_e, new_best = scored[0]
        # accept if better or with small probability
        if new_e <= best_e or random.random() < np.exp((best_e - new_e) / max(1e-6, cfg.temperature)):
            best, best_e = new_best, new_e
            history.append(best.copy())
    return best, history
