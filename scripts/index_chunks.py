import argparse
import logging
import time
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.core.log import setup_logging
from app.ingestion.models import Chunk
from app.storage.qdrant_store import QdrantStore
from scripts.embed_chunks import PROCESSED_DIR, STRATEGIES, file_hash, is_up_to_date

logger = logging.getLogger(__name__)

UPLOAD_STEP = 5000


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f]


def index_strategy(strategy: str, recreate: bool) -> None:
    settings = get_settings()
    source = PROCESSED_DIR / f"chunks_{strategy}.jsonl"
    cache = PROCESSED_DIR / f"embeddings_{strategy}.npz"
    if not cache.exists():
        logger.warning("%s not found, skipping. Run first: python -m scripts.embed_chunks", cache.name)
        return
    if not is_up_to_date(cache, settings.embedding_model, file_hash(source)):
        raise SystemExit(f"{cache.name} is stale (chunks or model changed). Run: python -m scripts.embed_chunks")

    chunks = read_chunks(source)
    with np.load(cache) as data:
        vectors = data["vectors"]

    store = QdrantStore(f"{settings.qdrant_collection}_{strategy}")
    store.ensure_collection(vectors.shape[1], recreate=recreate)

    started = time.perf_counter()
    for start in range(0, len(chunks), UPLOAD_STEP):
        end = start + UPLOAD_STEP
        store.upsert(chunks[start:end], vectors[start:end])
        logger.info("%s: %d/%d uploaded", store.collection, min(end, len(chunks)), len(chunks))
    logger.info(
        "%s: done in %.0fs, %d points in collection",
        store.collection, time.perf_counter() - started, store.count(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload cached chunk embeddings to Qdrant")
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the collections")
    args = parser.parse_args()

    setup_logging()
    for strategy in STRATEGIES:
        index_strategy(strategy, args.recreate)


if __name__ == "__main__":
    main()
