from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Iterable
import numpy as np


@dataclass
class Grid:
    h: int
    w: int
    fill: str = "⬛"
    data: np.ndarray = field(init=False)

    def __post_init__(self):
        self.data = np.full((self.h, self.w), self.fill, dtype=object)

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> "Grid":
        lines = list(lines)
        h = len(lines)
        w = max(len(line) for line in lines)
        g = cls(h, w)
        for r, line in enumerate(lines):
            cells = list(line)
            for c, ch in enumerate(cells):
                g.data[r, c] = ch
        return g

    def copy(self) -> "Grid":
        g = Grid(self.h, self.w, self.fill)
        g.data = self.data.copy()
        return g

    def put(self, r: int, c: int, ch: str):
        self.data[r, c] = ch

    def get(self, r: int, c: int) -> str:
        return self.data[r, c]

    def fill_rect(self, r0: int, c0: int, r1: int, c1: int, ch: str):
        self.data[r0:r1, c0:c1] = ch

    def as_lines(self) -> List[str]:
        return ["".join(self.data[r, :].tolist()) for r in range(self.h)]

    def __str__(self):
        return "\n".join(self.as_lines())

    def positions(self):
        for r in range(self.h):
            for c in range(self.w):
                yield r, c

    def neighbors4(self, r: int, c: int):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.h and 0 <= cc < self.w:
                yield rr, cc

    def to_image(self, cell_px: int = 32, pad: int = 8):
        """Render to a Pillow image (emoji as text)."""
        from PIL import Image, ImageDraw, ImageFont

        # Use a default font; emoji rendering varies by platform.
        img_w = self.w * cell_px + 2 * pad
        img_h = self.h * cell_px + 2 * pad
        img = Image.new("RGB", (img_w, img_h), (24, 24, 24))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("AppleColorEmoji.ttc", size=cell_px - 2)
        except Exception:
            # Fallback to default truetype
            font = ImageFont.load_default()
        for r in range(self.h):
            for c in range(self.w):
                ch = self.data[r, c]
                x = pad + c * cell_px
                y = pad + r * cell_px
                d.text((x, y), ch, fill=(240, 240, 240), font=font)
        return img
