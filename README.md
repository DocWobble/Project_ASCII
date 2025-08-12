# SeDiT — Semiotic Diffusion for Text

SeDiT is a prototype of **semiotic text diffusion**: a generator that composes a 2D grid of emoji “signs” and then compiles that slate into natural language.

This repo gives you a working scaffold:
- a discrete **grid state** over an emoji palette,
- simple **geometry energies** (connectedness, symmetry, bounds),
- a pluggable **listener energy** (LLM caption likelihood or keyword anchor heuristic),
- a tiny **diffusion/search** loop that edits cells to reduce energy,
- a placeholder **compiler** that verbalizes the grid.

> Goal: something you can hand to an autonomous coding agent to flesh out. It runs out of the box without internet; with an LLM key, it can use a listener for semantics.

## CLI

Run `python -m sedit.cli --prompt "pig" --size 12 --steps 12` to generate an
emoji slate. The command logs a breakdown of each energy term at every step and
saves intermediate PNGs, a `trajectory.gif`, an `energies.csv` file, and the
final `final.png` and `final.txt` outputs. Adjust semantic guidance with
`--listener-weight`.
