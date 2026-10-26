import logging
import re
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.ingestion.models import Document

logger = logging.getLogger(__name__)

EXTRA_SPACES = re.compile(r"[ \t]+")
EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    lines = (EXTRA_SPACES.sub(" ", line).strip() for line in text.splitlines())
    return EXTRA_BLANK_LINES.sub("\n\n", "\n".join(lines))


def _file_metadata(reader: PdfReader) -> dict:
    info = reader.metadata
    if info is None:
        return {}
    metadata = {"author": info.author, "pdf_title": (info.title or "").strip()}
    if info.creation_date:
        metadata["created_at"] = info.creation_date.isoformat()
    return {key: value for key, value in metadata.items() if value}


def parse_pdf(path: Path) -> Iterator[Document]:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            reader.decrypt("")
        pages = reader.pages
        total_pages = len(pages)
    except PdfReadError as e:
        logger.warning("%s skipped: cannot read PDF (%s)", path.name, e)
        return

    file_metadata = _file_metadata(reader)
    title = file_metadata.get("pdf_title") or path.stem
    empty_pages = 0

    for page_number, page in enumerate(pages, start=1):
        try:
            yield Document(
                title=title,
                text=_clean(page.extract_text(extraction_mode="layout") or ""),
                source=path.name,
                file_format="pdf",
                metadata={"page": page_number, "total_pages": total_pages, **file_metadata},
            )
        except ValidationError:
            empty_pages += 1

    if empty_pages:
        logger.warning("%s: %d of %d pages had no text (scanned images?)", path.name, empty_pages, total_pages)
