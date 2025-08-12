# CODING INSTRUCTIONS (for an autonomous coding agent)

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
