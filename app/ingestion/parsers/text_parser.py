import logging
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from app.ingestion.models import Document

logger = logging.getLogger(__name__)

MD_HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("%s is not valid UTF-8, unreadable characters replaced", path.name)
        text = raw.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n")


def _find_title(text: str, path: Path) -> str:
    if path.suffix.lower() == ".md":
        match = MD_HEADING.search(text)
        if match:
            return match.group(1)
    return path.stem


def parse_text(path: Path) -> Iterator[Document]:
    text = _read_text(path)
    file_format = "md" if path.suffix.lower() == ".md" else "txt"
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    try:
        yield Document(
            title=_find_title(text, path),
            text=text,
            source=path.name,
            file_format=file_format,
            metadata={"modified_at": modified.isoformat()},
        )
    except ValidationError as e:
        logger.warning("%s skipped: %s", path.name, e.errors()[0]["msg"])
