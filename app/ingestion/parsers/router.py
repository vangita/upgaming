from collections.abc import Callable, Iterator
from pathlib import Path

from app.ingestion.models import Document
from app.ingestion.parsers.docx_parser import parse_docx
from app.ingestion.parsers.jsonl_parser import parse_jsonl
from app.ingestion.parsers.pdf_parser import parse_pdf
from app.ingestion.parsers.text_parser import parse_text

Parser = Callable[[Path], Iterator[Document]]

PARSERS: dict[str, Parser] = {
    ".jsonl": parse_jsonl,
    ".txt": parse_text,
    ".md": parse_text,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
}

SUPPORTED_EXTENSIONS = tuple(PARSERS)


class UnsupportedFormatError(ValueError):
    pass


def parse_file(path: Path) -> Iterator[Document]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        raise UnsupportedFormatError(
            f"'{path.suffix}' is not supported. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    return parser(path)
