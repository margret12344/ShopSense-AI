import re
import html


def clean_text(text):

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\b[\w.-]+@[\w.-]+\.\w+\b",
        " ",
        text
    )

    # Convert to lowercase
    text = text.lower()

    # Keep letters, numbers, and useful punctuation
    text = re.sub(
        r"[^a-z0-9\s&/+._'-]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text