import html
import re

CAPTION_ARTIFACTS = re.compile(
    r"\[(?:Music|Applause|Laughter|Silence|Inaudible|Foreign)\]",
    re.IGNORECASE,
)


def clean_caption_text(text: str) -> str:
    text = html.unescape(text)
    text = CAPTION_ARTIFACTS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_at_sentence(text: str, max_chars: int = 300) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]
