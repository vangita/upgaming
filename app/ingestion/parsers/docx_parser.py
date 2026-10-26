import logging
from collections.abc import Iterator
from pathlib import Path

import docx
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph
from pydantic import ValidationError

from app.ingestion.models import Document

logger = logging.getLogger(__name__)


def _paragraph_to_text(paragraph: Paragraph) -> str:
    text = paragraph.text.strip()
    style = paragraph.style.name if paragraph.style else ""
    if text and style.startswith("Heading "):
        level = style.removeprefix("Heading ")
        if level.isdigit():
            return "#" * int(level) + " " + text
    if text and style == "Title":
        return "# " + text
    return text


def _table_to_text(table: Table) -> str:
    rows = (" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows)
    return "\n".join(rows)


def _extract_text(document) -> str:
    blocks = []
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            text = _paragraph_to_text(block)
        else:
            text = _table_to_text(block)
        if text:
            blocks.append(text)
    return "\n\n".join(blocks)


def _file_metadata(document) -> dict:
    props = document.core_properties
    metadata = {
        "author": props.author,
        "created_at": props.created.isoformat() if props.created else None,
        "modified_at": props.modified.isoformat() if props.modified else None,
    }
    return {key: value for key, value in metadata.items() if value}


def _find_title(document, text: str, path: Path) -> str:
    if document.core_properties.title:
        return document.core_properties.title
    for line in text.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ")
    return path.stem


def parse_docx(path: Path) -> Iterator[Document]:
    try:
        document = docx.Document(str(path))
    except PackageNotFoundError:
        logger.warning("%s skipped: not a valid DOCX file", path.name)
        return

    text = _extract_text(document)
    try:
        yield Document(
            title=_find_title(document, text, path),
            text=text,
            source=path.name,
            file_format="docx",
            metadata=_file_metadata(document),
        )
    except ValidationError as e:
        logger.warning("%s skipped: %s", path.name, e.errors()[0]["msg"])
