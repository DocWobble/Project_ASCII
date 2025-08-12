# Create a minimal repo skeleton for "SeDiT" (Semiotic Diffusion for Text)
import os, json, textwrap, zipfile, io, sys, pathlib

root = "/mnt/data/sedit"
paths = [
    "src/sedit",
    "scripts",
    "tests",
    "configs",
]

for p in paths:
    os.makedirs(os.path.join(root, p), exist_ok=True)

# Files content
readme = """# SeDiT — Semiotic Diffusion for Text

SeDiT is a prototype of **semiotic text diffusion**: a generator that composes a 2D grid of emoji “signs” and then compiles that slate into natural language.

This repo gives you a working scaffold:
- a discrete **grid state** over an emoji palette,
- simple **geometry energies** (connectedness, symmetry, bounds),
- a pluggable **listener energy** (LLM caption likelihood or keyword anchor heuristic),
- a tiny **diffusion/search** loop that edits cells to reduce energy,
- a placeholder **compiler** that verbalizes the grid.

> Goal: something you can hand to an autonomous coding agent to flesh out. It runs out of the box without internet; with an LLM key, it can use a listener for semantics.
"""

coding_instructions = """# CODING INSTRUCTIONS (for an autonomous coding agent)

You are operating inside this repository. Your objective is to turn this scaffold into a functional prototype of **Semiotic Diffusion for Text (SeDiT)**.

## Milestones
1. **MVP (Day 1–2):** Given a short prompt (e.g., "a yellow fish", "a pig"), produce a 12x12 emoji slate that a heuristic listener labels correctly. Write to `artifacts/` and show intermediate frames.
2. **Listener v1 (Day 2–3):** Add an LLM-based listener (OpenAI or any equivalent), scoring how well the grid conveys the prompt. Fall back to heuristics if no key is present.
3. **Compiler v1 (Day 3–4):** Transduce the final grid into a concise English sentence or bullet list. Keep it deterministic, rule-based to start.
4. **Training loop (Day 4–5):** Implement self-corruption and denoising training of a small neural policy (`sedit/denoiser.py`) to replace random search.
5. **Benchmarks (Day 5–6):** Listener exactness, geometry F1, edit cost. Add `pytest` tests and a `make benchmark` target.
6. **Docs + demo (Day 6):** A `scripts/demo.py` that runs end-to-end and saves a GIF of diffusion steps.

## Constraints & Rules
- **Language:** Python 3.10+.
- **No heavy dependencies** initially; use `numpy`, `networkx`, `regex`, `tqdm`, `pydantic` if helpful. Use `torch` only for the neural denoiser step (optional in MVP).
- The code must run **without internet** (heuristic listener on by default). If internet/keys are present, enable LLM listener automatically.
- Keep modules under `src/sedit/`. Use `ruff`/`black` formatting; add `pre-commit` hooks.
- Maintain cross-platform CLI (`python -m sedit.cli ...`).

## Tasks
- [ ] Implement grid ops and serialization (`sedit/grid.py`, `sedit/dsl.py`).
- [ ] Geometry energies: connectedness, bilateral symmetry, bounding ellipse roundness (`sedit/energy.py`).
- [ ] Listener interface with two backends: `HeuristicListener`, `LLMListener` (`sedit/listener.py`).
- [ ] Diffusion/search kernel with annealing and k-patch proposals (`sedit/denoiser.py`).
- [ ] Palette and anchors (`sedit/palette.py`): define ~512 emojis and anchor sets for common prompts.
- [ ] Compiler (`sedit/compiler.py`): deterministic verbalizer using anchors and simple templates.
- [ ] CLI (`sedit/cli.py`): `sedit generate --prompt "a pig" --size 12 --steps 12`.
- [ ] Demo script saving PNGs/GIFs (`scripts/demo.py`).
- [ ] Unit tests in `tests/` for grid ops and energies.
- [ ] Add a lightweight `torch`-free policy (tabular Q or logits per token) trained with self-corruption; upgrade to `torch` later.

## Acceptance Tests
- `pytest -q` passes.
- `python -m sedit.cli generate --prompt "pig" --size 12` writes an evolving slate and a final PNG under `artifacts/`.
- With `OPENAI_API_KEY` set, the listener leverages the API and improves slate quality vs. heuristic baseline on 20 prompts.

## Notes
- Keep the emoji grid **strictly fixed-size** during diffusion (default 12x12). Avoid ragged newlines; we want geometry invariants.
- Diffusion steps can be parallelized by proposing multiple edits and selecting the best under the energy.
- Document all magic numbers in `configs/default.yaml` and expose them via CLI flags.
"""

requirements = """numpy
networkx
regex
tqdm
Pillow
pydantic
"""

setup_cfg = """[flake8]
max-line-length = 100

[tool:pytest]
addopts = -q
"""

license_txt = """MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
..."""

init_py = """__all__ = ["grid", "palette", "energy", "listener", "denoiser", "compiler", "dsl", "cli"]
"""

grid_py = r'''from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Iterable, Optional
import numpy as np

@dataclass
class Grid:
    h: int
    w: int
    fill: str = "⬛"
    data: np.ndarray = field(init=False)

    def __post_init__(self):
        self.data = np.full((self.h, self.w), self.fill, dtype=object)

    @classmethod
    def from_lines(cls, lines: List[str]) -> "Grid":
        h = len(lines)
        w = max(len(line) for line in lines)
        g = cls(h, w, fill="⬛")
        for r, line in enumerate(lines):
            cells = list(line)
            for c, ch in enumerate(cells):
                g.data[r, c] = ch
        return g

    def copy(self) -> "Grid":
        g = Grid(self.h, self.w, self.fill)
        g.data = self.data.copy()
        return g

    def put(self, r: int, c: int, ch: str):
        self.data[r, c] = ch

    def get(self, r: int, c: int) -> str:
        return self.data[r, c]

    def fill_rect(self, r0: int, c0: int, r1: int, c1: int, ch: str):
        self.data[r0:r1, c0:c1] = ch

    def as_lines(self) -> List[str]:
        return ["".join(self.data[r, :].tolist()) for r in range(self.h)]

    def __str__(self):
        return "\n".join(self.as_lines())

    def positions(self):
        for r in range(self.h):
            for c in range(self.w):
                yield r, c

    def neighbors4(self, r: int, c: int):
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            rr, cc = r+dr, c+dc
            if 0 <= rr < self.h and 0 <= cc < self.w:
                yield rr, cc

    def to_image(self, cell_px: int = 32, pad: int = 8):
        """Render to a Pillow image (emoji as text)."""
        from PIL import Image, ImageDraw, ImageFont
        # Use a default font; emoji rendering varies by platform.
        img_w = self.w * cell_px + 2*pad
        img_h = self.h * cell_px + 2*pad
        img = Image.new("RGB", (img_w, img_h), (24,24,24))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("AppleColorEmoji.ttc", size=cell_px-2)
        except Exception:
            # Fallback to default truetype
            font = ImageFont.load_default()
        for r in range(self.h):
            for c in range(self.w):
                ch = self.data[r, c]
                x = pad + c * cell_px
                y = pad + r * cell_px
                d.text((x, y), ch, fill=(240,240,240), font=font)
        return img
'''

palette_py = r'''# Minimal emoji palette with background + anchors
BACKGROUND = "⬛"
BODY = "🟫"
WHITE = "⬜"
WATER = "🌊"
YELLOW = "🟡"
EYE = "👁️"
SNEUTRAL = "🙂"
PIG_NOSE = "🐽"
FISH = "🐟"
PIG = "🐷"

# Expandable: the agent should grow this to ~512 symbols and add codon logic.
PALETTE = [
    BACKGROUND, BODY, WHITE, WATER, YELLOW, EYE, PIG_NOSE, FISH, PIG
]

# Simple anchors for a few demo prompts
ANCHORS = {
    "pig": [EYE, PIG_NOSE, PIG],
    "fish": [FISH, YELLOW, WATER],
    "rocket": ["🚀", "🟥", "🟦", "🟨"],
}
'''

energy_py = r'''from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import networkx as nx
from .grid import Grid
from .palette import BACKGROUND

def connectedness_energy(g: Grid, foreground: List[str]) -> float:
    """Lower is better: 0 if a single connected component of non-background exists."""
    mask = np.isin(g.data, foreground)
    visited = np.zeros_like(mask, dtype=bool)
    comps = 0
    for r in range(g.h):
        for c in range(g.w):
            if mask[r,c] and not visited[r,c]:
                comps += 1
                # BFS
                stack = [(r,c)]
                visited[r,c] = True
                while stack:
                    rr, cc = stack.pop()
                    for nr, nc in ((rr+1,cc),(rr-1,cc),(rr,cc+1),(rr,cc-1)):
                        if 0 <= nr < g.h and 0 <= nc < g.w and mask[nr,nc] and not visited[nr,nc]:
                            visited[nr,nc] = True
                            stack.append((nr,nc))
    return max(0, comps-1)

def symmetry_energy(g: Grid, foreground: List[str]) -> float:
    """Horizontal bilateral symmetry cost."""
    mask = np.isin(g.data, foreground).astype(int)
    flipped = np.fliplr(mask)
    return float(np.sum(np.abs(mask - flipped)))/ (g.h * g.w)

def density_energy(g: Grid, foreground: List[str], target_fill=0.35) -> float:
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
'''

listener_py = r'''from __future__ import annotations
from typing import Optional
import os, re
from .grid import Grid
from .palette import ANCHORS

class HeuristicListener:
    """Keyword/anchor-based listener: cheap and offline."""
    def score(self, prompt: str, grid: Grid) -> float:
        prompt = prompt.lower()
        keys = []
        for k in ANCHORS:
            if k in prompt:
                keys.append(k)
        if not keys:
            # neutral score
            return 0.0
        score = 0.0
        text = "".join(grid.as_lines())
        for k in keys:
            for a in ANCHORS[k]:
                score += text.count(a)
        return score

class LLMListener:
    """Placeholder for an LLM-based listener. Implement using OpenAI or any LLM API."""
    def __init__(self):
        self.available = bool(os.environ.get("OPENAI_API_KEY"))

    def score(self, prompt: str, grid: Grid) -> float:
        if not self.available:
            return 0.0
        # TODO: Implement: caption the grid (serialized) and compute semantic similarity to prompt.
        # For now, return 0.0; the autonomous agent should fill this in.
        return 0.0
'''

denoiser_py = r'''from __future__ import annotations
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
        if new_ch == g.get(r,c):
            continue
        gg = g.copy()
        gg.put(r, c, new_ch)
        outs.append(gg)
    return outs

def run_diffusion(prompt: str, g: Grid, cfg: DiffusionConfig, energy_fn: Callable[..., Dict[str,float]]):
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
'''

compiler_py = r'''from __future__ import annotations
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
'''

dsl_py = r'''from __future__ import annotations
from typing import List
from .grid import Grid

def apply_ops(g: Grid, ops: List[str]):
    """Very small DSL: 'PUT r c ch' or 'FILL r0 c0 r1 c1 ch'"""
    for op in ops:
        parts = op.strip().split()
        if not parts:
            continue
        if parts[0].upper() == "PUT" and len(parts) >= 4:
            r, c = int(parts[1]), int(parts[2])
            ch = parts[3]
            g.put(r, c, ch)
        elif parts[0].upper() == "FILL" and len(parts) >= 6:
            r0, c0, r1, c1 = map(int, parts[1:5])
            ch = parts[5]
            g.fill_rect(r0, c0, r1, c1, ch)
'''

cli_py = r'''from __future__ import annotations
import argparse, os
from pathlib import Path
from .grid import Grid
from .palette import BACKGROUND, ANCHORS
from .energy import total_energy
from .denoiser import run_diffusion, DiffusionConfig
from .compiler import compile_to_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--outdir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    g = Grid(args.size, args.size, fill=BACKGROUND)

    anchors = []
    for k in ANCHORS:
        if k in args.prompt.lower():
            anchors = ANCHORS[k]

    cfg = DiffusionConfig(steps=args.steps, anchors=anchors, seed=args.seed)

    best, history = run_diffusion(args.prompt, g, cfg, energy_fn=total_energy)

    # Save frames
    for i, frame in enumerate(history):
        img = frame.to_image(cell_px=28, pad=6)
        img.save(os.path.join(args.outdir, f"frame_{i:02d}.png"))
    best.to_image(cell_px=28, pad=6).save(os.path.join(args.outdir, "final.png"))

    # Compile to text
    text = compile_to_text(args.prompt, best)
    with open(os.path.join(args.outdir, "final.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(text)

if __name__ == "__main__":
    main()
'''

demo_py = r'''from pathlib import Path
import os
from sedit.cli import main

if __name__ == "__main__":
    # Convenience wrapper; identical to calling the CLI.
    main()
'''

test_grid = r'''from sedit.grid import Grid

def test_grid_put_get():
    g = Grid(3,3, fill="⬛")
    g.put(1,1,"🐽")
    assert g.get(1,1) == "🐽"
    assert len(g.as_lines()) == 3
'''

default_yaml = r'''size: 12
steps: 12
proposals_per_step: 64
temperature: 0.8
'''

# Write files
files = {
    "README.md": readme,
    "CODING_INSTRUCTIONS.md": coding_instructions,
    "requirements.txt": requirements,
    "setup.cfg": setup_cfg,
    "LICENSE": license_txt,
    "src/sedit/__init__.py": init_py,
    "src/sedit/grid.py": grid_py,
    "src/sedit/palette.py": palette_py,
    "src/sedit/energy.py": energy_py,
    "src/sedit/listener.py": listener_py,
    "src/sedit/denoiser.py": denoiser_py,
    "src/sedit/compiler.py": compiler_py,
    "src/sedit/dsl.py": dsl_py,
    "src/sedit/cli.py": cli_py,
    "scripts/demo.py": demo_py,
    "tests/test_grid.py": test_grid,
    "configs/default.yaml": default_yaml,
}

for rel, content in files.items():
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# Create a zip for easy download
zip_path = "/mnt/data/SeDiT_skeleton.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            arc = os.path.relpath(full, root)
            z.write(full, arcname=f"sedit/{arc}")

zip_path

