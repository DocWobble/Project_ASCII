from statistics import median
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sedit.grid import Grid
from sedit.denoiser import run_diffusion, DiffusionConfig
from sedit.listener import HeuristicListener


def _score(prompt: str, weight: float) -> float:
    g = Grid(6, 6)
    listener = HeuristicListener()
    cfg = DiffusionConfig(
        steps=8, proposals_per_step=32, seed=0, listener_weight=weight
    )
    best, _, _ = run_diffusion(prompt, g, cfg, listener=listener)
    return listener.score(prompt, best)


def test_listener_weight_improves_median():
    prompts = ["pig", "fish", "sun", "car"]
    baseline = [_score(p, 0.0) for p in prompts]
    weighted = [_score(p, 1.0) for p in prompts]
    assert median(weighted) >= median(baseline)

