import re

from app.ingestion.models import Chunk, Document

WORD = re.compile(r"\S+")


def chunk_fixed(document: Document, chunk_size: int = 150, overlap: int = 30) -> list[Chunk]:
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = list(WORD.finditer(document.text))
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        chunks.append(Chunk.from_document(
            document,
            index=len(chunks),
            char_start=window[0].start(),
            char_end=window[-1].end(),
        ))
        if start + chunk_size >= len(words):
            break

    return chunks
