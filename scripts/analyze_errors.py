"""Error analysis of the answer evaluation: why each wrong answer is wrong.

For every answer the judge did not grade correct, it rebuilds what the LLM read and sorts the error:
- retrieval: a gold document never reached the LLM, so the answer could not be complete;
- generation: the LLM read every gold document and still answered wrong, or abstained.
Hybrid search gives the same results on the same index, so the sources are rebuilt without LLM calls:
the method's first search plus the agent's follow-up searches saved with each answer.
It also lists correct answers whose citations miss a gold document or point to another document.
Results are saved to data/eval/results/error_analysis.json.
"""
import inspect
import json
import logging
from collections import Counter
from pathlib import Path

from app.core.config import get_settings
from app.core.log import setup_logging
from app.evaluation.metrics import citation_accuracy, citation_coverage, document_title
from app.rag.agent import InvestigationAgent
from app.rag.pipeline import RAGPipeline
from app.rag.tools import SourceList
from app.retrieval.retriever import Retriever
from app.storage.qdrant_store import QdrantStore, SearchHit

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
QUESTIONS_FILE = DATA_DIR / "eval" / "questions.jsonl"
RESULTS_DIR = DATA_DIR / "eval" / "results"
OUTPUT_FILE = RESULTS_DIR / "error_analysis.json"


def default(function, name: str):
    return inspect.signature(function).parameters[name].default


# The search depths the app uses, read from the code so the analysis follows any change to them
SIMPLE_TOP_K = default(RAGPipeline.answer, "top_k")
AGENT_TOP_K = default(InvestigationAgent, "top_k")
AGENT_SEARCH_K = default(InvestigationAgent, "search_k")

# Problems of the questions themselves, found by reading the failed answers
NOTES = {
    "hp-multihop4": "Label noise: the HotpotQA question is garbled ('Luz Pfannenstiel' for Lutz Pfannenstiel) "
                    "and its gold answer is a phrase ('the best known globetrotters')",
}


def sources_read(retriever: Retriever, method: str, record: dict) -> list[SearchHit]:
    if method == "simple":
        return retriever.search(record["question"], limit=SIMPLE_TOP_K)
    # The agent's own bookkeeping: a chunk found again by a follow-up search is not new
    sources = SourceList()
    sources.add(retriever.search(record["question"], limit=AGENT_TOP_K))
    for query in record["searches"]:
        if query:
            sources.add(retriever.search(query, limit=AGENT_SEARCH_K))
    return sources.hits


def classify(record: dict, question: dict, read: list[str]) -> tuple[str, str]:
    if "error" in record:
        return "llm_error", record["error"]
    if question["category"] == "unanswerable":
        return "generation", "answered a question the documents cannot answer"
    missing = sorted(set(question["gold_docs"]) - set(read))
    if missing:
        return "retrieval", "never read: " + ", ".join(missing)
    if not record["answered"]:
        return "generation", "abstained although it read every gold document"
    if question["category"] == "contradictory":
        if record["verdict"] == "partial":
            return "generation", "read both versions, said they differ, but gave one"
        return "generation", "read both versions but gave one, without a conflict"
    return "generation", "read every gold document, but the answer is wrong or incomplete"


def citation_issue(record: dict, question: dict) -> dict | None:
    """A correct answer whose citations miss a gold document or point to another document."""
    gold = set(question["gold_docs"])
    if not gold or not record["answered"]:
        return None
    accuracy, coverage = citation_accuracy(record["cited"], gold), citation_coverage(record["cited"], gold)
    if coverage == 1 and accuracy == 1:
        return None
    return {
        "id": record["id"], "category": record["category"],
        "not_cited": sorted(gold - set(record["cited"])),
        "cited_other": sorted({title for title in record["cited"] if title not in gold}),
    }


def analyze_run(path: Path, questions: dict[str, dict], retriever: Retriever) -> dict:
    run = path.stem.removeprefix("answers_")
    method = run.split("_")[0]
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    failures, citation_issues = [], []
    for record in records:
        question = questions.get(record["id"])
        if question is None:
            continue
        if record["verdict"] == "correct":
            if issue := citation_issue(record, question):
                citation_issues.append(issue)
            continue
        hits = sources_read(retriever, method, record)
        if len(hits) != record["sources_read"]:
            logger.warning("%s %s: rebuilt %d sources, the run read %d", run, record["id"], len(hits), record["sources_read"])
        gold = set(question["gold_docs"])
        kind, detail = classify(record, question, [document_title(hit, gold) for hit in hits])
        failures.append({
            "id": record["id"], "category": record["category"], "verdict": record["verdict"],
            "type": kind, "detail": detail, "searches": record["searches"], "answer": record["answer"],
            "note": NOTES.get(record["id"]),
        })
    return {
        "run": run, "answers": len(records),
        "failure_types": dict(Counter(f["type"] for f in failures)),
        "failures": failures, "citation_issues": citation_issues,
    }


def print_run(result: dict) -> None:
    types = ", ".join(f"{kind} {count}" for kind, count in result["failure_types"].items()) or "none"
    print(f"\n## {result['run']}: {len(result['failures'])} of {result['answers']} answers not correct ({types})")
    print("| id | category | verdict | type | detail |\n|---|---|---|---|---|")
    for f in result["failures"]:
        detail = f["detail"] + (f" (searched: {'; '.join(f['searches'])})" if f["searches"] else " (no follow-up search)")
        if f["note"]:
            detail += f". {f['note']}"
        print(f"| {f['id']} | {f['category']} | {f['verdict']} | {f['type']} | {detail} |")
    print(f"\nCorrect answers with incomplete citations: {len(result['citation_issues'])}")
    for issue in result["citation_issues"]:
        parts = [f"not cited: {', '.join(issue['not_cited'])}"] if issue["not_cited"] else []
        parts += [f"also cited: {', '.join(issue['cited_other'])}"] if issue["cited_other"] else []
        print(f"- {issue['id']} ({issue['category']}): {'; '.join(parts)}")


def main() -> None:
    setup_logging()
    questions = {q["id"]: q for q in (json.loads(line) for line in QUESTIONS_FILE.open(encoding="utf-8"))}
    retriever = Retriever(QdrantStore(get_settings().main_collection))
    results = [analyze_run(path, questions, retriever) for path in sorted(RESULTS_DIR.glob("answers_*.jsonl"))]
    OUTPUT_FILE.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    for result in results:
        print_run(result)
    logger.info("Saved to %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
