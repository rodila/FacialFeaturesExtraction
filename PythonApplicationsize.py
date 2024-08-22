import nltk
from nltk.tokenize import word_tokenize

# Define size synonyms for jawline, nose, and lips
jawline_sizes = {
    "small": ["small", "petite", "delicate"],
    "medium": ["average", "normal", "typical"],
    "large": ["large", "big", "prominent", "pronunced"]
}

nose_sizes = {
    "small": ["small", "petite", "tiny"],
    "medium": ["average", "normal", "typical"],
    "large": ["large", "big", "prominent"]
}

lips_sizes = {
    "small": ["small", "petite", "tiny"],
    "medium": ["average", "normal", "typical"],
    "large": ["large", "big", "full"]
}

def extract_size_keywords(text):
    # Tokenize the input text
    tokens = word_tokenize(text.lower())

    # Initialize variables to store extracted size keywords
    jawline = None
    nose = None
    lips = None

    # Find the position of "jawline", "nose", and "lips" in the text
    for i, token in enumerate(tokens):
        if token == "jawline":
            jawline_index = i
        elif token == "nose":
            nose_index = i
        elif token == "lips":
            lips_index = i

    # Extract size keywords preceding "jawline", "nose", and "lips"
    if "jawline" in locals() and jawline_index > 0:
        for size, synonyms in jawline_sizes.items():
            if any(synonym in tokens[jawline_index - 1] for synonym in synonyms):
                jawline = size
                break

    if "nose" in locals() and nose_index > 0:
        for size, synonyms in nose_sizes.items():
            if any(synonym in tokens[nose_index - 1] for synonym in synonyms):
                nose = size
                break

    if "lips" in locals() and lips_index > 0:
        for size, synonyms in lips_sizes.items():
            if any(synonym in tokens[lips_index - 1] for synonym in synonyms):
                lips = size
                break

    return jawline, nose, lips



