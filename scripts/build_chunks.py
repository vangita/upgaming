import logging
import statistics
from collections.abc import Callable
from pathlib import Path

from app.core.log import setup_logging
from app.ingestion.chunking.fixed import chunk_fixed
from app.ingestion.chunking.structured import chunk_structured
from app.ingestion.dedup import deduplicate
from app.ingestion.models import Chunk, Document
from app.ingestion.parsers.router import parse_file

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_FILE = DATA_DIR / "raw" / "hotpotqa_paragraphs.jsonl"
PROCESSED_DIR = DATA_DIR / "processed"

STRATEGIES: dict[str, Callable[[Document], list[Chunk]]] = {
    "fixed": chunk_fixed,
    "structured": chunk_structured,
}


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def log_stats(name: str, chunks: list[Chunk]) -> None:
    sizes = [len(chunk.text.split()) for chunk in chunks]
    logger.info(
        "%-10s %6d chunks | words: mean %.0f, median %.0f, max %d",
        name, len(chunks), statistics.mean(sizes), statistics.median(sizes), max(sizes),
    )


def main() -> None:
    setup_logging()
    if not RAW_FILE.exists():
        raise SystemExit(f"{RAW_FILE} not found. Run first: python -m scripts.prepare_dataset")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    documents = deduplicate(parse_file(RAW_FILE))

    for name, chunk_document in STRATEGIES.items():
        chunks = [chunk for document in documents for chunk in chunk_document(document)]
        output = PROCESSED_DIR / f"chunks_{name}.jsonl"
        write_chunks(output, chunks)
        log_stats(name, chunks)
        logger.info("Saved to %s", output)


if __name__ == "__main__":
    main()
