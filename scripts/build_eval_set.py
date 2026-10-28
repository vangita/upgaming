"""Builds the evaluation set data/eval/questions.jsonl: 47 questions in five categories.

- multi_hop and multi_doc: HotpotQA validation questions sampled with a fixed seed ("bridge" questions
  need a chain of two paragraphs, "comparison" questions compare two entities). Sampled questions whose
  gold answer does not answer the question are left out (DROPPED, with the reason).
- The other questions are written by hand: about the INC-2041 sample documents (data/samples/), with
  single-paragraph questions and unanswerable ones about the HotpotQA paragraphs.

Each question has gold documents - titles as the retriever returns them (chunk.title) - for Recall@K
and MRR. Unanswerable questions have no answer and no gold documents: the expected result is "not
enough information". Contradictory questions expect both versions of the fact and a reported conflict.
"""
import json
import logging
import random
from collections import Counter
from pathlib import Path

from app.core.log import setup_logging

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
QUESTIONS_FILE = DATA_DIR / "raw" / "hotpotqa_questions.jsonl"
OUTPUT_FILE = DATA_DIR / "eval" / "questions.jsonl"

SEED = 2026
SAMPLES = [("bridge", "multi_hop", 12), ("comparison", "multi_doc", 10)]
DROPPED = {
    "5a7d27f05542995ed0d1660d": "asks for a country, the gold answer is 'Iranian'",
    "5a7f1c195542993067513633": "the question is garbled, so 'no' cannot be checked",
    "5a876f785542993e715abf42": "asks which person, the gold answer is a list of roles",
}

# Titles of the sample documents, as the parsers produce them
POSTMORTEM = "INC-2041 Post-mortem: card payments outage"
STATUS_PAGE = "inc-2041-status-page"
SUPPORT = "INC-2041 Customer support summary"
REPORT = "Reliability report Q1 2026"

CAUSE_CONFLICT = (
    "Sources disagree on the cause: an expired TLS certificate on the internal payment gateway "
    "(post-mortem, Q1 report) vs a database failover in the primary data center (status page)."
)

# (id, category, question, answer, gold documents); answer None = unanswerable
MANUAL = [
    # Simple: one document holds the answer
    ("inc-s1", "simple", "Which error code did declined card transactions return during INC-2041?",
     "GW-503", [POSTMORTEM]),
    ("inc-s2", "simple", "How many INC-2041 support tickets came in by phone?", "245", [SUPPORT]),
    ("inc-s3", "simple", "How many merchants qualified for the transaction fee refund after INC-2041?",
     "214", [SUPPORT]),
    ("inc-s4", "simple", "Who owns the INC-2041 action item to alert before certificates expire, "
     "and when is it due?", "The SRE team, due 4 April 2026", [POSTMORTEM]),
    ("inc-s5", "simple", "What was the availability of card payments in Q1 2026?",
     "99.94%", [REPORT]),
    ("hp-s1", "simple", "Which Canadian Football League team does quarterback Dan LeFevour play for?",
     "Winnipeg Blue Bombers", ["Dan LeFevour"]),
    ("hp-s2", "simple", "Which band was Stephen Christian the lead vocalist of?", "Anberlin",
     ["Stephen Christian"]),
    ("hp-s3", "simple", "Where was the 1945 VFL Grand Final held?", "Princes Park in Melbourne",
     ["1945 VFL Grand Final"]),
    ("hp-s4", "simple", "Which Soviet fighter is the Chengdu J-7 a license-built version of?",
     "The Mikoyan-Gurevich MiG-21", ["Chengdu J-7"]),
    ("hp-s5", "simple", "When was the actress Julia Ling born?", "February 14, 1983", ["Julia Ling"]),
    # Multi-hop: a fact from one document is needed to use another
    ("inc-mh1", "multi_hop", "How many minutes after the TLS certificate expired did the first customer "
     "report about failed card payments arrive?",
     "7 minutes: the certificate expired at 14:05 UTC (post-mortem), the first customer report "
     "arrived at 14:12 UTC (support summary)", [POSTMORTEM, SUPPORT]),
    # Multi-doc: facts from several documents are combined
    ("inc-md1", "multi_doc", "Which payment methods were not affected by INC-2041, and was one of them "
     "affected by another incident in Q1 2026?",
     "Bank transfers and wallet payments were not affected; wallet payments were unavailable for "
     "22 minutes during INC-2017", [POSTMORTEM, REPORT]),
    ("inc-md2", "multi_doc", "Which action items from the INC-2041 post-mortem also appear among the "
     "goals for Q2 2026 in the reliability report?",
     "Alerts for expiring certificates and a status page update within 10 minutes of detection",
     [POSTMORTEM, REPORT]),
    ("inc-md3", "multi_doc", "Merchants complained about the status page during INC-2041. How late was "
     "its first update, and which action item addresses this?",
     "It was first updated 15 minutes after the failures began; the action item requires a status "
     "page update within 10 minutes of detection (owner: Incident management)", [SUPPORT, POSTMORTEM]),
    # Unanswerable: no document holds the answer (checked against the paragraphs and the samples)
    ("inc-u1", "unanswerable", "What was the financial cost of INC-2041 for Corvane Pay?", None, []),
    ("inc-u2", "unanswerable", "What is the name of the on-call engineer who was paged during INC-2041?",
     None, []),
    ("inc-u3", "unanswerable", "How many card transactions were declined in total during INC-2041?",
     None, []),
    ("inc-u4", "unanswerable", "Who made the faulty configuration change that caused INC-2017?", None, []),
    # Carice van Houten and Halina Reijn appear in the corpus, but not these facts about them
    ("hp-u1", "unanswerable", "When was Carice van Houten born?", None, []),
    ("hp-u2", "unanswerable", "How many Golden Globe Awards has Halina Reijn won?", None, []),
    ("hp-u3", "unanswerable", "Who won the 2030 FIFA World Cup?", None, []),
    ("hp-u4", "unanswerable", "What was the closing price of Apple stock on 3 March 2026?", None, []),
    # Contradictory: the sample documents disagree on purpose (see scripts/make_samples.py)
    ("inc-c1", "contradictory", "Which system changes caused the payment processing disruption in March, "
     "which services were affected, and which documents confirm this?",
     "Card payments were affected. " + CAUSE_CONFLICT, [POSTMORTEM, STATUS_PAGE, REPORT]),
    ("inc-c2", "contradictory", "What caused the card payments outage on 14 March 2026?", CAUSE_CONFLICT,
     [POSTMORTEM, STATUS_PAGE, REPORT]),
    ("inc-c3", "contradictory", "Was the INC-2041 outage caused by a database problem?", CAUSE_CONFLICT,
     [POSTMORTEM, STATUS_PAGE, REPORT]),
    ("inc-c4", "contradictory", "How long did the INC-2041 card payments outage last?",
     "Sources disagree: 47 minutes (post-mortem, Q1 report) vs about 65 minutes (status page)",
     [POSTMORTEM, STATUS_PAGE, REPORT]),
    ("inc-c5", "contradictory", "At what time was INC-2041 resolved?",
     "Sources disagree: 14:52 UTC (post-mortem) vs 15:10 UTC (status page)", [POSTMORTEM, STATUS_PAGE]),
    ("inc-c6", "contradictory", "What share of card transactions failed during INC-2041?",
     "Sources disagree: about 12% (post-mortem) vs 18% (Q1 report)", [POSTMORTEM, REPORT]),
]


def question(id_: str, category: str, text: str, answer: str | None, gold_docs: list[str], origin: str) -> dict:
    return {
        "id": id_, "category": category, "question": text, "answer": answer,
        "gold_docs": gold_docs, "origin": origin,
    }


def sample_hotpotqa() -> list[dict]:
    hotpotqa = [json.loads(line) for line in QUESTIONS_FILE.open(encoding="utf-8")]
    # One generator for both samples, in a fixed order, so the same questions come out on every run
    rng = random.Random(SEED)
    questions = []
    for hotpotqa_type, category, size in SAMPLES:
        sampled = rng.sample([q for q in hotpotqa if q["type"] == hotpotqa_type], size)
        kept = [q for q in sampled if q["id"] not in DROPPED]
        for number, q in enumerate(kept, start=1):
            gold_docs = list(dict.fromkeys(fact["title"] for fact in q["supporting_facts"]))
            questions.append(question(
                f"hp-{category.replace('_', '')}{number}", category, q["question"], q["answer"],
                gold_docs, f"hotpotqa:{q['id']}",
            ))
    return questions


def main() -> None:
    setup_logging()
    if not QUESTIONS_FILE.exists():
        raise SystemExit(f"{QUESTIONS_FILE} not found. Run first: python -m scripts.prepare_dataset")

    questions = [question(*item, origin="manual") for item in MANUAL] + sample_hotpotqa()
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "question ids must be unique"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    counts = Counter(q["category"] for q in questions)
    logger.info("%d questions saved to %s: %s", len(questions), OUTPUT_FILE, dict(counts))


if __name__ == "__main__":
    main()
