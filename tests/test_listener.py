from sedit.grid import Grid
from sedit.palette import BACKGROUND, FISH, ANCHORS
from sedit.listener import HeuristicListener
from sedit.energy import total_energy


def test_listener_energy_reward():
    g = Grid(2, 2, fill=BACKGROUND)
    g.put(0, 0, FISH)
    listener = HeuristicListener()
    anchors = ANCHORS["fish"]
    fg = anchors
    e = total_energy("fish", g, anchors, fg, listener, listener_weight=1.0)
    assert e["listener"] < 0
    e0 = total_energy("fish", g, anchors, fg, listener, listener_weight=0.0)
    assert e0["listener"] == 0
