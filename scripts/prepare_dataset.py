import json
import logging
from pathlib import Path

from datasets import load_dataset

from app.core.log import setup_logging

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PARAGRAPHS_FILE = RAW_DIR / "hotpotqa_paragraphs.jsonl"
QUESTIONS_FILE = RAW_DIR / "hotpotqa_questions.jsonl"


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_paragraphs(dataset) -> list[dict]:
    paragraphs = []
    for example in dataset:
        context = example["context"]
        for title, sentences in zip(context["title"], context["sentences"]):
            paragraphs.append({"title": title, "sentences": sentences})
    return paragraphs


def extract_questions(dataset) -> list[dict]:
    questions = []
    for example in dataset:
        facts = example["supporting_facts"]
        questions.append({
            "id": example["id"],
            "question": example["question"],
            "answer": example["answer"],
            "type": example["type"],
            "supporting_facts": [
                {"title": title, "sent_id": sent_id}
                for title, sent_id in zip(facts["title"], facts["sent_id"])
            ],
        })
    return questions


def main() -> None:
    setup_logging()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading HotpotQA (distractor, validation)...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    paragraphs = extract_paragraphs(dataset)
    write_jsonl(PARAGRAPHS_FILE, paragraphs)
    logger.info("Saved %d paragraphs to %s", len(paragraphs), PARAGRAPHS_FILE)

    questions = extract_questions(dataset)
    write_jsonl(QUESTIONS_FILE, questions)
    logger.info("Saved %d questions to %s", len(questions), QUESTIONS_FILE)


if __name__ == "__main__":
    main()
