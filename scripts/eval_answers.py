"""Answer evaluation: runs each RAG method on the evaluation set and grades its answers.

1. Answers: every question goes through the method as the app runs it (hybrid search), with the app's
   LLM or another one (--model). The answer, cited documents, conflicts, LLM calls, tokens and time
   are appended at once to data/eval/results/answers_<method>_<model>.jsonl. A new run continues
   where the last one stopped, so the daily rate limit only pauses the evaluation. Delete the file to
   answer again from scratch (e.g. after a prompt change).
2. Grading: the LLM judge (another model with its own rate limits) grades answers without a verdict;
   --regrade grades all of them again (e.g. after a change to the judge or to a reference answer).
3. Metrics of every results file, per question category -> data/eval/results/answers_summary.json.

Groq's free tier gives each model 8K tokens per minute and 200K per day: the script keeps under the
per-minute limit and waits when it is hit anyway; when the daily one is reached it stops and keeps
what it has.
"""
import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from app.core.config import get_settings
from app.core.log import setup_logging
from app.evaluation.judge import JUDGE_MODEL, SCORES, AnswerJudge, grade_answer
from app.evaluation.metrics import citation_accuracy, citation_coverage, document_title, mean
from app.llm.client import DailyLimitError, LLMClient, LLMError, MinuteLimitError
from app.rag.agent import InvestigationAgent
from app.rag.pipeline import RAGPipeline
from app.retrieval.retriever import Retriever
from app.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
QUESTIONS_FILE = DATA_DIR / "eval" / "questions.jsonl"
RESULTS_DIR = DATA_DIR / "eval" / "results"
SUMMARY_FILE = RESULTS_DIR / "answers_summary.json"

METHODS: dict[str, Callable[..., InvestigationAgent | RAGPipeline]] = {
    "agent": InvestigationAgent,
    "simple": RAGPipeline,
}
GROUPS = ("all", "simple", "multi_hop", "multi_doc", "unanswerable", "contradictory")
METRICS = (
    "answer_correctness", "citation_accuracy", "citation_coverage", "abstained_correctly",
    "false_abstention", "conflict_detected", "spurious_conflict", "llm_calls", "tokens", "seconds",
)

# Under Groq's 8K tokens per minute, with a margin: a question's tokens are only known after it ran
TOKENS_PER_MINUTE = 7000
RATE_LIMIT_WAIT = 60
RETRIES = 2

T = TypeVar("T")


def with_retries(call: Callable[[], T]) -> T:
    # The SDK already retries a 429 a few times within seconds; a longer wait outlasts a per-minute limit.
    # A DailyLimitError is not retried: it stops the run
    for _ in range(RETRIES):
        try:
            return call()
        except MinuteLimitError as e:
            logger.warning("%s, waiting %d s", e, RATE_LIMIT_WAIT)
            time.sleep(RATE_LIMIT_WAIT)
    return call()


def run_name(method: str, model: str) -> str:
    # "agent" + "openai/gpt-oss-120b" -> "agent_gpt-oss-120b"
    return f"{method}_{model.split('/')[-1]}"


def results_file(name: str) -> Path:
    return RESULTS_DIR / f"answers_{name}.jsonl"


def load_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = (json.loads(line) for line in path.open(encoding="utf-8"))
    return {record["id"]: record for record in records}


def save_records(path: Path, records: dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def answer_question(method: InvestigationAgent | RAGPipeline, llm: LLMClient, question: dict) -> dict:
    before = replace(llm.usage)
    started = time.perf_counter()
    record = {"id": question["id"], "category": question["category"], "question": question["question"]}
    try:
        result = with_retries(lambda: method.answer(question["question"]))
    except (MinuteLimitError, DailyLimitError):
        raise
    except LLMError as e:
        # Any other LLM failure is a failed answer, as a user of the app would see it
        logger.warning("%s failed: %s", question["id"], e)
        record |= {"answer": "", "answered": False, "cited": [], "invalid_citations": [], "conflicts": [],
                   "searches": [], "sources_read": 0, "error": str(e), "verdict": "incorrect", "reason": str(e)}
    else:
        gold = set(question["gold_docs"])
        record |= {
            "answer": result.answer,
            "answered": result.answered,
            "cited": [document_title(citation.hit, gold) for citation in result.citations],
            "invalid_citations": result.invalid_citations,
            "conflicts": [conflict.description for conflict in result.conflicts],
            "searches": result.searches,
            "sources_read": len(result.sources),
            "verdict": None,
            "reason": None,
        }
    tokens = llm.usage.prompt_tokens + llm.usage.completion_tokens - before.prompt_tokens - before.completion_tokens
    record |= {
        "llm_calls": llm.usage.calls - before.calls,
        "tokens": tokens,
        "seconds": round(time.perf_counter() - started, 1),
    }
    return record


def run_answers(name: str, method: InvestigationAgent | RAGPipeline, questions: list[dict]) -> bool:
    """Answers the questions that have no saved answer; False if the rate limit stopped the run."""
    path = results_file(name)
    records = load_records(path)
    todo = [q for q in questions if q["id"] not in records]
    logger.info("%s: %d questions answered before, %d to go", name, len(questions) - len(todo), len(todo))

    for number, question in enumerate(todo, start=1):
        try:
            record = answer_question(method, method.llm, question)
        except LLMError as e:
            logger.error("%s stopped at %s: %s. Run the script again later to continue.", name, question["id"], e)
            return False
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "%s %d/%d %s: answered=%s, %d LLM calls, %d tokens, %.1f s",
            name, number, len(todo), question["id"], record["answered"],
            record["llm_calls"], record["tokens"], record["seconds"],
        )
        time.sleep(max(0.0, record["tokens"] / TOKENS_PER_MINUTE * 60 - record["seconds"]))
    return True


def run_judge(name: str, questions: dict[str, dict], judge: AnswerJudge, regrade: bool = False) -> None:
    path = results_file(name)
    records = load_records(path)
    # Failed answers keep their verdict: they were not graded by the judge
    todo = [
        r for r in records.values()
        if r["id"] in questions and (r["verdict"] is None or (regrade and "error" not in r))
    ]
    for record in todo:
        question = questions[record["id"]]
        try:
            verdict = with_retries(lambda: grade_answer(
                judge, question, record["answer"], record["answered"], record["conflicts"],
            ))
        except LLMError as e:
            logger.error("Grading stopped: %s. Run the script again later to continue.", e)
            break
        record["verdict"], record["reason"] = verdict.label, verdict.reason
    save_records(path, records)
    logger.info("%s: %d answers graded", name, len(todo))


def record_metrics(record: dict, question: dict) -> dict:
    """The metrics of one answer; None where a metric does not apply to the question."""
    gold = set(question["gold_docs"])
    answerable = record["category"] != "unanswerable"
    contradictory = record["category"] == "contradictory"
    abstained = float(not record["answered"])
    found_conflict = float(bool(record["conflicts"]))
    # Citations are checked only where there is something to cite: an answer to a question with gold documents
    checked = bool(gold) and record["answered"]
    return {
        "answer_correctness": SCORES.get(record["verdict"]),
        "citation_accuracy": citation_accuracy(record["cited"], gold) if checked else None,
        "citation_coverage": citation_coverage(record["cited"], gold) if checked else None,
        "abstained_correctly": None if answerable else abstained,
        "false_abstention": abstained if answerable else None,
        "conflict_detected": found_conflict if contradictory else None,
        "spurious_conflict": None if contradictory else found_conflict,
        "llm_calls": record["llm_calls"],
        "tokens": record["tokens"],
        "seconds": record["seconds"],
    }


def summarize(path: Path, questions: dict[str, dict]) -> list[dict]:
    records = [r for r in load_records(path).values() if r["id"] in questions]
    rows = []
    for group in GROUPS:
        selected = [r for r in records if group in ("all", r["category"])]
        if not selected:
            continue
        metrics = [record_metrics(r, questions[r["id"]]) for r in selected]
        rows.append({
            "run": path.stem.removeprefix("answers_"), "group": group, "questions": len(selected),
            # Not graded yet, or the judge's reply could not be read: left out of answer_correctness
            "ungraded": sum(SCORES.get(r["verdict"]) is None for r in selected),
            **{name: mean([m[name] for m in metrics]) for name in METRICS},
        })
    return rows


def print_tables(rows: list[dict]) -> None:
    def cell(name: str, value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:.0f}" if name == "tokens" else f"{value:.1f}" if name in ("llm_calls", "seconds") else f"{value:.2f}"

    for group in GROUPS:
        group_rows = [row for row in rows if row["group"] == group]
        if not group_rows:
            continue
        print(f"\nquestions: {group}\n| run | n | ungraded | " + " | ".join(METRICS) + " |")
        print("|---" * (len(METRICS) + 3) + "|")
        for row in group_rows:
            values = " | ".join(cell(name, row[name]) for name in METRICS)
            print(f"| {row['run']} | {row['questions']} | {row['ungraded']} | {values} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--model", help="the answering LLM (default: the app's GROQ_MODEL)")
    parser.add_argument("--only", nargs="+", metavar="ID", help="only these question ids, e.g. for a pilot run")
    parser.add_argument("--regrade", action="store_true", help="grade the saved answers again")
    args = parser.parse_args()
    setup_logging()

    questions = [json.loads(line) for line in QUESTIONS_FILE.open(encoding="utf-8")]
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]
    by_id = {q["id"]: q for q in questions}
    settings = get_settings()
    model = args.model or settings.groq_model
    api_key = settings.groq_api_key.get_secret_value()
    retriever = Retriever(QdrantStore(settings.main_collection))
    # No fallback model, unlike the app: a run must measure the one model in its name
    llm, judge = LLMClient(api_key, model), AnswerJudge(LLMClient(api_key, JUDGE_MODEL))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        name = run_name(method, model)
        # The judge has its own rate limits, so the answers so far are graded even after a stop
        finished = run_answers(name, METHODS[method](retriever, llm=llm), questions)
        run_judge(name, by_id, judge, regrade=args.regrade)
        if not finished:
            break

    # Every run so far, also those of other models, so the summary always compares all of them
    rows = [row for path in sorted(RESULTS_DIR.glob("answers_*.jsonl")) for row in summarize(path, by_id)]
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge_model": JUDGE_MODEL, "search_mode": "hybrid", "questions": len(questions), "rows": rows,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print_tables(rows)
    logger.info("Summary saved to %s", SUMMARY_FILE)


if __name__ == "__main__":
    main()
