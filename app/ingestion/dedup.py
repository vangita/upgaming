import hashlib
import logging
import re
from collections.abc import Iterable

from app.ingestion.models import Document

logger = logging.getLogger(__name__)

WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip().lower()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def deduplicate(documents: Iterable[Document]) -> list[Document]:
    unique: dict[str, Document] = {}
    duplicates = 0

    for document in documents:
        key = content_hash(document.text)
        kept = unique.get(key)

        if kept is None:
            document.metadata["content_hash"] = key
            unique[key] = document
            continue

        duplicates += 1
        aliases = kept.metadata.get("aliases", [])
        if document.title != kept.title and document.title not in aliases:
            kept.metadata["aliases"] = [*aliases, document.title]

    logger.info("Deduplication: %d unique, %d duplicates removed", len(unique), duplicates)
    return list(unique.values())
