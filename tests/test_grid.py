import os
import sys

# Allow running tests without installing the package by adding the project root
# (which contains a ``sedit`` symlink) to ``sys.path``.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sedit.grid import Grid


def test_grid_put_get():
    g = Grid(3, 3, fill="⬛")
    g.put(1, 1, "🐽")
    assert g.get(1, 1) == "🐽"
    assert len(g.as_lines()) == 3
