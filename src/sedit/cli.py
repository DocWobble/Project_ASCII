from __future__ import annotations
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
