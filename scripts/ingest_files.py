import argparse
import logging
from pathlib import Path

from app.core.config import get_settings
from app.core.log import setup_logging
from app.ingestion.chunking.fixed import chunk_fixed
from app.ingestion.chunking.structured import chunk_structured
from app.ingestion.parsers.router import SUPPORTED_EXTENSIONS
from app.ingestion.service import IngestionError, IngestionService
from app.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# The app answers from the structured collection; the fixed one exists to compare the two strategies
CHUNKERS = {"structured": chunk_structured, "fixed": chunk_fixed}


def collect_files(paths: list[Path]) -> list[Path]:
    # A folder means every supported file directly inside it
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS))
        else:
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Add, update or delete files in a Qdrant collection")
    parser.add_argument("paths", nargs="*", type=Path, help="files or folders to index")
    parser.add_argument("--delete", nargs="+", default=[], metavar="NAME", help="file names to remove")
    parser.add_argument(
        "--chunking", choices=CHUNKERS, default="structured",
        help="chunking strategy and its collection (default: structured, the one the app uses)",
    )
    args = parser.parse_args()

    setup_logging()
    store = QdrantStore(f"{get_settings().qdrant_collection}_{args.chunking}")
    service = IngestionService(store, chunker=CHUNKERS[args.chunking])
    store.ensure_collection(service.embedder.dimension)

    for name in args.delete:
        service.delete(name)
    for path in collect_files(args.paths):
        try:
            result = service.ingest_file(path)
        except (IngestionError, FileNotFoundError) as e:
            logger.warning("%s skipped: %s", path.name, e)
            continue
        if result.duplicates:
            logger.warning("%s: already indexed from other files: %s", path.name, ", ".join(result.duplicates))
    logger.info("%s: %d points", store.collection, store.count())


if __name__ == "__main__":
    main()
