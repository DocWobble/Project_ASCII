import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sedit.cli import run_prompt


def test_best_total_decreases(tmp_path):
    run_prompt(
        "pig",
        size=6,
        steps=15,
        outdir=tmp_path,
        seed=0,
        listener_weight=1.0,
        use_llm_listener=False,
    )

    with open(tmp_path / "energies.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    best = [float(r["best_total"]) for r in rows]
    decreases = sum(
        1 for i in range(1, len(best)) if best[i] < min(best[:i])
    )
    assert decreases >= 0.7 * (len(best) - 1)

