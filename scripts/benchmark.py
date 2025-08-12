from __future__ import annotations
"""Benchmark utility comparing heuristic and LLM listeners on a prompt suite.

This script runs each prompt twice – once with the heuristic listener and once
with the LLM listener – saving outputs under ``<outdir>/<prompt>/<listener>``.
It then writes a Markdown report with captions, listener deltas and an embedded
energy curve as well as an ``index.html`` gallery of final images.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sedit.cli import run_prompt, _slugify
from sedit.palette import ANCHORS


def _plot_curve(vals: List[float], path: Path) -> None:
    """Draw a simple line plot using Pillow."""
    if not vals:
        return
    w, h = 120, 60
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        vmax = vmin + 1.0
    pts = []
    for i, v in enumerate(vals):
        x = i * (w - 1) / max(1, len(vals) - 1)
        y = h - 1 - (v - vmin) / (vmax - vmin) * (h - 1)
        pts.append((x, y))
    draw.line(pts, fill="black", width=1)
    img.save(path)


def run_suite(outdir: Path, size: int, steps: int, listener_weight: float, use_llm: bool) -> None:
    prompts = list(ANCHORS.keys())[:20]
    rows = []

    for prompt in prompts:
        slug = _slugify(prompt)
        for mode, use_llm_listener in (("heuristic", False), ("llm", use_llm)):
            run_dir = outdir / slug / mode
            res = run_prompt(
                prompt,
                size,
                steps,
                run_dir,
                listener_weight=listener_weight,
                use_llm_listener=use_llm_listener,
            )
            # plot energy
            energies = [e["best_total"] for e in res["energies"]]
            _plot_curve(energies, run_dir / "energy.png")

        # collect captions and deltas
        heur_cap = (outdir / slug / "heuristic" / "final.txt").read_text().strip()
        llm_cap = (outdir / slug / "llm" / "final.txt").read_text().strip()
        energies = list(csv.DictReader(open(outdir / slug / "llm" / "energies.csv")))
        if energies:
            w = listener_weight or 1.0
            start = -float(energies[0]["listener"]) / w
            end = -float(energies[-1]["listener"]) / w
            delta = end - start
        else:
            delta = 0.0
        rows.append((prompt, heur_cap, llm_cap, delta, slug))

    # write report
    with open(outdir / "report.md", "w", encoding="utf-8") as f:
        f.write("|prompt|heuristic|llm|listener Δ|energy|\n")
        f.write("|-|-|-|-|-|\n")
        for prompt, hc, lc, delta, slug in rows:
            f.write(
                f"|{prompt}|{hc} [img]({slug}/heuristic/final.png) [gif]({slug}/heuristic/trajectory.gif) "
                f"|{lc} [img]({slug}/llm/final.png) [gif]({slug}/llm/trajectory.gif) "
                f"|{delta:.2f}|![]({slug}/llm/energy.png)|\n"
            )

    # write gallery
    with open(outdir / "index.html", "w", encoding="utf-8") as f:
        f.write("<html><body><h1>SeDiT Benchmark</h1><div style='display:flex;flex-wrap:wrap;'>")
        for prompt in prompts:
            slug = _slugify(prompt)
            img_path = f"{slug}/llm/final.png"
            f.write(
                f"<div style='margin:4px;text-align:center'><a href='{slug}/llm/final.png'>"
                f"<img src='{img_path}' width='64'><br>{prompt}</a></div>"
            )
        f.write("</div></body></html>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="bench")
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--listener-weight", type=float, default=1.0)
    parser.add_argument("--use-llm-listener", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_suite(outdir, args.size, args.steps, args.listener_weight, args.use_llm_listener)


if __name__ == "__main__":
    main()

