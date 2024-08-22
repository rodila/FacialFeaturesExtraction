import nltk
from nltk.tokenize import word_tokenize

# Define color synonyms for hair, skin, and eyes
hair_colors = {
    "brown": ["brown"],
    "blonde": ["blonde", "gold", "golden"],
    "black": ["black", "dark"],
    "red": ["red", "ginger"]
}

skin_colors = {
    "light": ["light", "fair", "pale", "white"],
    "medium": ["medium", "olive", "tan"],
    "dark": ["dark", "deep", "black", "bronze"]
}

eye_colors = {
    "blue": ["blue", "baby-blue"],
    "green": ["green"],
    "brown": ["brown"],
    "hazel": ["hazel", "bronze"],
    "gray": ["gray"]
}

def extract_color_keywords(text):
    # Tokenize the input text
    tokens = word_tokenize(text.lower())

    # Initialize variables to store extracted color keywords
    hair = None
    skin = None
    eyes = None

    # Extract color keywords for hair
    for i, token in enumerate(tokens):
        if token == "hair":
            if i > 0:
                for color, synonyms in hair_colors.items():
                    if any(synonym in tokens[i - 1] for synonym in synonyms):
                        hair = color
                        break

    # Extract color keywords for skin
    for i, token in enumerate(tokens):
        if token == "skin":
            if i > 0:
                for color, synonyms in skin_colors.items():
                    if any(synonym in tokens[i - 1] for synonym in synonyms):
                        skin = color
                        break

    # Extract color keywords for eyes
    for i, token in enumerate(tokens):
        if token == "eyes":
            if i > 0:
                for color, synonyms in eye_colors.items():
                    if any(synonym in tokens[i - 1] for synonym in synonyms):
                        eyes = color
                        break

    return hair, skin, eyes


