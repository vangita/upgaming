"""Retrieval evaluation: Recall@K and MRR of every search mode on both chunking strategies.

It makes no LLM calls, so it is free and fast enough for a larger question set:
- the evaluation set (data/eval/questions.jsonl) without its unanswerable questions, per category;
- a random sample of HotpotQA validation questions (--hotpotqa N): 39 questions are too few to show
  small differences between configurations.
K counts documents: the top chunks are turned into a document ranking first (ranked_documents).
Results are saved to data/eval/results/retrieval.json.
"""
import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

from app.core.config import get_settings
from app.core.log import setup_logging
from app.embeddings.embedder import get_embedder
from app.evaluation.metrics import document_title, mean, ranked_documents, recall_at_k, reciprocal_rank
from app.retrieval.retriever import Retriever, SearchMode
from app.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
QUESTIONS_FILE = DATA_DIR / "eval" / "questions.jsonl"
HOTPOTQA_FILE = DATA_DIR / "raw" / "hotpotqa_questions.jsonl"
RESULTS_FILE = DATA_DIR / "eval" / "results" / "retrieval.json"

STRATEGIES = ("structured", "fixed")
MODES: tuple[SearchMode, ...] = ("semantic", "keyword", "hybrid")
TOP_CHUNKS = 10
KS = (1, 3, 5, 10)
SEED = 42


def load_eval_set() -> list[dict]:
    questions = [json.loads(line) for line in QUESTIONS_FILE.open(encoding="utf-8")]
    return [q for q in questions if q["gold_docs"]]


def load_hotpotqa_sample(size: int) -> list[dict]:
    hotpotqa = [json.loads(line) for line in HOTPOTQA_FILE.open(encoding="utf-8")]
    return [
        {
            "id": q["id"], "category": "hotpotqa", "question": q["question"],
            "gold_docs": list(dict.fromkeys(fact["title"] for fact in q["supporting_facts"])),
        }
        for q in random.Random(SEED).sample(hotpotqa, size)
    ]


def evaluate(retriever: Retriever, mode: SearchMode, question: dict) -> dict:
    gold = set(question["gold_docs"])
    started = time.perf_counter()
    hits = retriever.search(question["question"], mode=mode, limit=TOP_CHUNKS)
    elapsed_ms = (time.perf_counter() - started) * 1000
    ranked = ranked_documents([document_title(hit, gold) for hit in hits])
    result = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in KS}
    return {"id": question["id"], **result, "rr": reciprocal_rank(ranked, gold), "ms": elapsed_ms, "ranked": ranked}


def summarize(results: list[dict]) -> dict:
    summary = {f"recall@{k}": mean([r[f"recall@{k}"] for r in results]) for k in KS}
    return {
        "questions": len(results), **summary,
        "mrr": mean([r["rr"] for r in results]),
        "ms": mean([r["ms"] for r in results]),
    }


def print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    columns = [f"recall@{k}" for k in KS] + ["mrr", "ms"]
    print(f"\n{title}\n| configuration | n | " + " | ".join(columns) + " |")
    print("|---" * (len(columns) + 2) + "|")
    for name, row in rows:
        values = " | ".join(f"{row[c]:.0f}" if c == "ms" else f"{row[c]:.3f}" for c in columns)
        print(f"| {name} | {row['questions']} | {values} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hotpotqa", type=int, default=500, help="size of the HotpotQA sample (0 = none)")
    args = parser.parse_args()
    setup_logging()

    groups = {"eval_set": load_eval_set()}
    if args.hotpotqa:
        groups[f"hotpotqa_{args.hotpotqa}"] = load_hotpotqa_sample(args.hotpotqa)
    settings = get_settings()
    embedder = get_embedder()
    # Loads the model before the timing starts, so the first query is not slower than the others
    embedder.embed_query("warm-up")

    summary, per_question = [], []
    for strategy in STRATEGIES:
        retriever = Retriever(QdrantStore(f"{settings.qdrant_collection}_{strategy}"), embedder)
        for mode in MODES:
            config = {"chunking": strategy, "mode": mode}
            for group, questions in groups.items():
                results = [evaluate(retriever, mode, q) for q in questions]
                summary.append({**config, "group": group, **summarize(results)})
                if group == "eval_set":
                    per_question += [{**config, **r} for r in results]
                    by_category = defaultdict(list)
                    for q, r in zip(questions, results):
                        by_category[q["category"]].append(r)
                    summary += [{**config, "group": c, **summarize(rs)} for c, rs in by_category.items()]
            logger.info("%s / %s done", strategy, mode)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    settings_used = {"top_chunks": TOP_CHUNKS, "ks": KS, "hotpotqa_sample": args.hotpotqa, "seed": SEED}
    RESULTS_FILE.write_text(
        json.dumps({"settings": settings_used, "summary": summary, "per_question": per_question}, indent=1),
        encoding="utf-8",
    )

    def rows(group: str, strategies=STRATEGIES) -> list[tuple[str, dict]]:
        return [(f"{s['chunking']} / {s['mode']}", s) for s in summary
                if s["group"] == group and s["chunking"] in strategies]

    for group in groups:
        print_table(f"{group}: all configurations", rows(group))
    for category in ("simple", "multi_hop", "multi_doc", "contradictory"):
        print_table(f"eval_set, {category}: structured chunking", rows(category, ("structured",)))
    logger.info("Results saved to %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
