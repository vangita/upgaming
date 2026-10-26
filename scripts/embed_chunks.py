import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.core.log import setup_logging
from app.embeddings.embedder import get_embedder

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
# structured first: it becomes the main index
STRATEGIES = ("structured", "fixed")


def file_hash(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def is_up_to_date(cache_path: Path, model_name: str, source_hash: str) -> bool:
    if not cache_path.exists():
        return False
    with np.load(cache_path) as cache:
        return str(cache["model"]) == model_name and str(cache["source_hash"]) == source_hash


def read_texts(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def main() -> None:
    setup_logging()
    model_name = get_settings().embedding_model

    for strategy in STRATEGIES:
        source = PROCESSED_DIR / f"chunks_{strategy}.jsonl"
        output = PROCESSED_DIR / f"embeddings_{strategy}.npz"
        if not source.exists():
            raise SystemExit(f"{source} not found. Run first: python -m scripts.build_chunks")

        source_hash = file_hash(source)
        if is_up_to_date(output, model_name, source_hash):
            logger.info("%s is up to date, skipping", output.name)
            continue

        texts = read_texts(source)
        logger.info("Embedding %d %s chunks...", len(texts), strategy)
        vectors = get_embedder().embed_documents(texts, show_progress=True)
        np.savez(output, vectors=vectors, model=model_name, source_hash=source_hash)
        logger.info("Saved %s vectors to %s", vectors.shape, output)


if __name__ == "__main__":
    main()
