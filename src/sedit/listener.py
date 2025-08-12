from __future__ import annotations
from typing import Optional
import os
from .grid import Grid
from .palette import ANCHORS


class HeuristicListener:
    """Keyword/anchor-based listener: cheap and offline."""

    def score(self, prompt: str, grid: Grid) -> float:
        prompt = prompt.lower()
        keys = []
        for k in ANCHORS:
            if k in prompt:
                keys.append(k)
        if not keys:
            # neutral score
            return 0.0
        score = 0.0
        text = "".join(grid.as_lines())
        for k in keys:
            for a in ANCHORS[k]:
                score += text.count(a)
        return score


class LLMListener:
    """Placeholder for an LLM-based listener. Implement using OpenAI or any LLM API."""

    def __init__(self):
        self.available = bool(os.environ.get("OPENAI_API_KEY"))

    def score(self, prompt: str, grid: Grid) -> float:
        if not self.available:
            return 0.0
        # TODO: Implement: caption the grid (serialized) and compute semantic similarity to prompt.
        # For now, return 0.0; the autonomous agent should fill this in.
        return 0.0
