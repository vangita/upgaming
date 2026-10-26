import re

from app.ingestion.chunking.fixed import WORD
from app.ingestion.models import Chunk, Document

Span = tuple[int, int]

HEADING = re.compile(r"^#{1,6}\s", re.MULTILINE)
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
SEPARATORS = [PARAGRAPH_BREAK, SENTENCE_END]


def _word_count(text: str, span: Span) -> int:
    return len(WORD.findall(text, *span))


def _trim(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_at(text: str, span: Span, pattern: re.Pattern) -> list[Span]:
    start, end = span
    cuts = [m.start() for m in pattern.finditer(text, start, end) if m.start() > start]
    bounds = [start, *cuts, end]
    pieces = (_trim(text, a, b) for a, b in zip(bounds, bounds[1:]))
    return [p for p in pieces if p[0] < p[1]]


def _split_words(text: str, span: Span, chunk_size: int) -> list[Span]:
    words = list(WORD.finditer(text, *span))
    return [
        (words[i].start(), words[min(i + chunk_size, len(words)) - 1].end())
        for i in range(0, len(words), chunk_size)
    ]


def _split_to_fit(text: str, span: Span, chunk_size: int, level: int = 0) -> list[Span]:
    if _word_count(text, span) <= chunk_size:
        return [span]
    if level == len(SEPARATORS):
        return _split_words(text, span, chunk_size)

    result = []
    for piece in _split_at(text, span, SEPARATORS[level]):
        result.extend(_split_to_fit(text, piece, chunk_size, level + 1))
    return result


def _merge(text: str, pieces: list[tuple[Span, int]], chunk_size: int, min_size: int) -> list[Span]:
    merged = []
    current = None
    current_words = 0
    current_section = None

    for (start, end), section in pieces:
        words = _word_count(text, (start, end))
        if current is not None:
            too_big = current_words + words > chunk_size
            new_section = section != current_section and current_words >= min_size
            if too_big or new_section:
                merged.append(current)
                current = None

        if current is None:
            current, current_words = (start, end), 0
        current = (current[0], end)
        current_words += words
        current_section = section

    if current is None:
        return merged
    if merged and current_words < min_size:
        merged[-1] = (merged[-1][0], current[1])
    else:
        merged.append(current)
    return merged


def chunk_structured(document: Document, chunk_size: int = 150, min_size: int = 30) -> list[Chunk]:
    text = document.text
    pieces = []
    for section_id, section in enumerate(_split_at(text, (0, len(text)), HEADING)):
        for span in _split_to_fit(text, section, chunk_size):
            pieces.append((span, section_id))

    spans = _merge(text, pieces, chunk_size, min_size)
    return [Chunk.from_document(document, i, start, end) for i, (start, end) in enumerate(spans)]
