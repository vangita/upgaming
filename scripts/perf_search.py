"""Performance test of the search at full dataset size (71,597 chunks), without LLM calls.

1. Latency: N HotpotQA questions, one after another, in every search mode, plus the query embedding
   alone: mean, p50, p95, p99 and the slowest search.
2. Load: the same hybrid searches from several threads at once, as concurrent API requests would
   run them: searches per second and latency under load.

The LLM is left out on purpose: its time depends on Groq, not on this system, and the free tier
allows only 200K tokens a day. Results are saved to data/eval/results/performance.json.
Run it against the local Qdrant to measure the system itself, not the network:
QDRANT_URL=http://localhost:6333 QDRANT_API_KEY= python -m scripts.perf_search
"""
import argparse
import json
import logging
import random
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from app.api.stats import percentile
from app.core.config import get_settings
from app.core.log import setup_logging
from app.embeddings.embedder import get_embedder
from app.retrieval.retriever import Retriever, SearchMode
from app.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HOTPOTQA_FILE = DATA_DIR / "raw" / "hotpotqa_questions.jsonl"
RESULTS_FILE = DATA_DIR / "eval" / "results" / "performance.json"

MODES: tuple[SearchMode, ...] = ("semantic", "keyword", "hybrid")
THREADS = (1, 4, 8, 16)
TOP_CHUNKS = 10
SEED = 7


def timed(call: Callable[[], object]) -> float:
    started = time.perf_counter()
    call()
    return (time.perf_counter() - started) * 1000


def summary(times_ms: list[float]) -> dict:
    values = sorted(times_ms)
    return {
        "n": len(values),
        "mean_ms": round(statistics.mean(values), 1),
        "p50_ms": round(percentile(values, 0.50), 1),
        "p95_ms": round(percentile(values, 0.95), 1),
        "p99_ms": round(percentile(values, 0.99), 1),
        "max_ms": round(values[-1], 1),
    }


def under_load(retriever: Retriever, queries: list[str], threads: int) -> dict:
    def search(query: str) -> float:
        return timed(lambda: retriever.search(query, mode="hybrid", limit=TOP_CHUNKS))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        times = list(pool.map(search, queries))
    elapsed = time.perf_counter() - started
    return {"threads": threads, "searches_per_s": round(len(queries) / elapsed, 1), **summary(times)}


def print_table(title: str, rows: list[dict], first: str) -> None:
    columns = [first, "n", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"]
    if "searches_per_s" in rows[0]:
        columns.insert(1, "searches_per_s")
    print(f"\n{title}\n| " + " | ".join(columns) + " |\n" + "|---" * len(columns) + "|")
    for row in rows:
        print("| " + " | ".join(str(row[column]) for column in columns) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", type=int, default=1000, help="questions for the latency test")
    parser.add_argument("--load-queries", type=int, default=400, help="questions per thread count in the load test")
    args = parser.parse_args()
    setup_logging()

    hotpotqa = [json.loads(line)["question"] for line in HOTPOTQA_FILE.open(encoding="utf-8")]
    queries = random.Random(SEED).sample(hotpotqa, args.queries)
    settings = get_settings()
    store = QdrantStore(settings.main_collection)
    embedder = get_embedder()
    retriever = Retriever(store, embedder)
    info = store.client.get_collection(store.collection)
    logger.info("Qdrant at %s, %d points", urlparse(settings.qdrant_url).hostname, info.points_count)

    # Warm-up: the first searches load caches and open connections
    for query in queries[:20]:
        retriever.search(query, mode="hybrid", limit=TOP_CHUNKS)

    latency = [{"step": "query embedding", **summary([timed(lambda: embedder.embed_query(q)) for q in queries])}]
    for mode in MODES:
        times = [timed(lambda: retriever.search(q, mode=mode, limit=TOP_CHUNKS)) for q in queries]
        latency.append({"step": f"{mode} search", **summary(times)})
        logger.info("%s: %d searches done", mode, len(times))
    load = [under_load(retriever, queries[:args.load_queries], threads) for threads in THREADS]

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({
        "settings": {"queries": args.queries, "load_queries": args.load_queries, "top_chunks": TOP_CHUNKS,
                     "seed": SEED, "qdrant_host": urlparse(settings.qdrant_url).hostname},
        "collection": {"name": store.collection, "points": info.points_count, "segments": info.segments_count},
        "latency": latency,
        "load": load,
    }, indent=1), encoding="utf-8")

    print_table(f"Latency, one search at a time ({info.points_count:,} chunks)", latency, "step")
    print_table("Hybrid search under load", load, "threads")
    logger.info("Results saved to %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
