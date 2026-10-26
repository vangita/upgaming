import json
import logging
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from app.ingestion.models import Document

logger = logging.getLogger(__name__)


def _record_to_document(record: dict, path: Path, line_number: int) -> Document:
    title = record.pop("title", None)

    if "text" in record:
        text = record.pop("text")
    elif "sentences" in record:
        text = "".join(record["sentences"])
    else:
        raise ValueError("record has neither 'text' nor 'sentences'")

    return Document(
        title=title,
        text=text,
        source=path.name,
        file_format="jsonl",
        metadata={"line": line_number, **record},
    )


def parse_jsonl(path: Path) -> Iterator[Document]:
    skipped = 0

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                yield _record_to_document(record, path, line_number)
            except (json.JSONDecodeError, ValueError, ValidationError) as e:
                skipped += 1
                logger.warning("%s:%d skipped: %s", path.name, line_number, e)

    if skipped:
        logger.warning("%s: %d lines skipped", path.name, skipped)
