# Minimal emoji palette with background + anchors
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

# Expandable: the agent should grow this to ~512 symbols and add codon logic.
PALETTE = [
    BACKGROUND, BODY, WHITE, WATER, YELLOW, EYE, PIG_NOSE, FISH, PIG
]

# Simple anchors for a few demo prompts
ANCHORS = {
    "pig": [EYE, PIG_NOSE, PIG],
    "fish": [FISH, YELLOW, WATER],
    "rocket": ["🚀", "🟥", "🟦", "🟨"],
}
