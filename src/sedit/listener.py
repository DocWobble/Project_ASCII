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
    """LLM-based listener using OpenAI's API.

    If an ``OPENAI_API_KEY`` is present and the ``openai`` package is installed,
    this listener will query the chat completions API to obtain a semantic
    similarity score between the prompt and the grid's emoji artwork.  The model
    is asked to output a single floating point number in ``[0,1]`` where higher
    means better alignment.  Any failure (missing package, network error, bad
    response) results in a neutral score of ``0.0``.
    """

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.available = False
        self.client = None
        if self.api_key:
            try:
                import openai  # type: ignore

                openai.api_key = self.api_key
                self.client = openai
                self.available = True
            except Exception:
                # Missing dependency or invalid key – fall back to heuristic listener.
                self.available = False

    def score(self, prompt: str, grid: Grid) -> float:
        if not self.available or self.client is None:
            return 0.0

        text = "\n".join(grid.as_lines())
        try:
            resp = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a critic scoring how well emoji art matches a prompt. "
                            "Respond with only a number between 0 and 1."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Prompt: {prompt}\nArt:\n{text}\nScore:",
                    },
                ],
                temperature=0.0,
                max_tokens=1,
            )
            out = resp["choices"][0]["message"]["content"].strip()
            return float(out)
        except Exception:
            # Network or parsing failure – treat as neutral.
            return 0.0
