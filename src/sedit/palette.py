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

PALETTE = sorted(
    {symbol for symbol in BASE_PALETTE + [s for v in ANCHORS.values() for s in v]}
)
