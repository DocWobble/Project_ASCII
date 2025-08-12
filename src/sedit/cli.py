from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List

from .compiler import compile_to_text
from .denoiser import DiffusionConfig, run_diffusion
from .energy import total_energy
from .grid import Grid
from .listener import HeuristicListener, LLMListener
from .palette import ANCHORS, BACKGROUND


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "prompt"


def run_prompt(
    prompt: str,
    size: int,
    steps: int,
    outdir: str | Path,
    seed: int = 0,
    listener_weight: float = 1.0,
    use_llm_listener: bool = False,
) -> Dict[str, object]:
    """Run diffusion for a single prompt and persist artifacts."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    g = Grid(size, size, fill=BACKGROUND)

    anchors: List[str] = []
    for k in ANCHORS:
        if k in prompt.lower():
            anchors = ANCHORS[k]

    cfg = DiffusionConfig(
        steps=steps, anchors=anchors, seed=seed, listener_weight=listener_weight
    )

    listener = HeuristicListener()
    if use_llm_listener:
        llm = LLMListener()
        if llm.available:
            listener = llm

    best, history, energies = run_diffusion(
        prompt, g, cfg, listener=listener, energy_fn=total_energy
    )

    # Save frames and collect for GIF
    images = []
    for i, frame in enumerate(history):
        img = frame.to_image(cell_px=28, pad=6)
        img.save(outdir / f"frame_{i:02d}.png")
        images.append(img)
    if images:
        images[0].save(
            outdir / "trajectory.gif",
            save_all=True,
            append_images=images[1:],
            duration=200,
            loop=0,
        )
    best.to_image(cell_px=28, pad=6).save(outdir / "final.png")

    # Save energy CSV
    if energies:
        fieldnames = ["step"] + list(energies[0].keys())
        with open(outdir / "energies.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, e in enumerate(energies):
                row = {"step": i}
                row.update(e)
                writer.writerow(row)

    # Compile to text
    text = compile_to_text(prompt, best)
    with open(outdir / "final.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")

    return {"grid": best, "energies": energies, "caption": text}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str)
    group.add_argument("--batch", type=str, help="file with one prompt per line")
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--outdir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--listener-weight", type=float, default=1.0)
    parser.add_argument("--use-llm-listener", action="store_true")
    args = parser.parse_args()

    if args.batch:
        with open(args.batch, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        for prompt in prompts:
            slug = _slugify(prompt)
            out = Path(args.outdir) / slug
            res = run_prompt(
                prompt,
                args.size,
                args.steps,
                out,
                seed=args.seed,
                listener_weight=args.listener_weight,
                use_llm_listener=args.use_llm_listener,
            )
            print(f"{prompt}: {res['caption']}")
    else:
        res = run_prompt(
            args.prompt,
            args.size,
            args.steps,
            args.outdir,
            seed=args.seed,
            listener_weight=args.listener_weight,
            use_llm_listener=args.use_llm_listener,
        )
        print(res["caption"])


if __name__ == "__main__":
    main()

