"""Palette and anchor definitions.

The palette starts with a small base set and then grows automatically from the
anchor symbols below.  This makes it easy to expand the demo to new prompts
without manually curating a global palette list.
"""

# Base colours / parts used across multiple prompts
BACKGROUND = "⬛"
BODY = "🟫"
WHITE = "⬜"
WATER = "🌊"
YELLOW = "🟡"
EYE = "👁️"
SNEUTRAL = "🙂"
PIG_NOSE = "🐽"
FISH = "🐟"
PIG = "🐷"

# Anchor sets for ~20 common demo prompts.  Each list contains a few emoji that
# should plausibly appear in grids describing the keyword.  These are used by
# the heuristic listener as well as the geometry/anchor energies.
ANCHORS = {
    "pig": [EYE, PIG_NOSE, PIG],
    "fish": [FISH, YELLOW, WATER],
    "rocket": ["🚀", "🟥", "🟦", "🟨"],
    "cat": ["🐱", "😺", "🐾"],
    "dog": ["🐶", "🐾", "🦴"],
    "tree": ["🌳", "🌲", "🍃"],
    "car": ["🚗", "🚙", "🛣️"],
    "house": ["🏠", "🏡", "🚪"],
    "boat": ["⛵", "🚤", WATER],
    "flower": ["🌸", "🌼", "🌺"],
    "bird": ["🐦", "🐤", "🐥"],
    "sun": ["☀️", "🌞", "😎"],
    "moon": ["🌙", "🌕", "⭐"],
    "star": ["⭐", "✨", "🌟"],
    "smile": ["😀", "🙂", "😊"],
    "heart": ["❤️", "💖", "💗"],
    "fire": ["🔥", "💥", "⚡"],
    "banana": ["🍌", "🐒", "🍌"],
    "cake": ["🎂", "🍰", "🕯️"],
    "book": ["📚", "📖", "🔖"],
}

# Build the full palette as the union of base symbols and all anchors
BASE_PALETTE = [
    BACKGROUND,
    BODY,
    WHITE,
    WATER,
    YELLOW,
    EYE,
    SNEUTRAL,
    PIG_NOSE,
    FISH,
    PIG,
]

PALETTE = sorted({symbol for symbol in BASE_PALETTE + [s for v in ANCHORS.values() for s in v]})
