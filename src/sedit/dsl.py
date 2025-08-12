from __future__ import annotations
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
