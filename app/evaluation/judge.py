"""Answer Correctness: an LLM judge compares the system answer with the reference answer.

Comparing strings does not work here: a comparison question ("Valley International or Aberdeen?")
names both candidates in every answer, right or wrong, and "two" vs "2" or a shorter form of a name
is still correct. The judge is a different model from the one that answers (no preference for its
own wording) and has its own Groq rate limits, so judging does not use up the answering model's
daily tokens.
"""
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from app.llm.client import LLMClient, Message
from app.rag.citations import normalize_answer, strip_citations

JUDGE_MODEL = "qwen/qwen3.8-27b"

Label = Literal["correct", "partial", "incorrect", "invalid"]
SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

REPLY_FORMAT = """Reply with one word on the first line: CORRECT, PARTIAL or INCORRECT.
On the second line, give a one-sentence reason."""

JUDGE_PROMPT = f"""You grade the answers of a question-answering system against a reference answer.
Judge only by the reference answer, not by your own knowledge of the topic.

- CORRECT: the system answer gives the same facts as the reference. Wording, extra details and
  formats (e.g. "two" or "2", a full or a short name) do not matter.
- PARTIAL: the reference has several parts and the system answer gets only some of them right.
- INCORRECT: the system answer gives a different fact or entity, contradicts the reference, or does
  not answer the question.

{REPLY_FORMAT}"""

# A separate prompt: with the general rules above the judge graded a one-version answer PARTIAL
# ("one of two parts") although it never said that the sources disagree
CONFLICT_PROMPT = f"""You grade the answers of a question-answering system against a reference answer.
The documents disagree on the fact the question asks about, and the reference answer gives both
versions. The system should notice the disagreement. Wording and formats do not matter.

- CORRECT: the system answer gives both versions.
- PARTIAL: the system answer says that the sources disagree or differ, but gives only one version.
- INCORRECT: the system answer gives only one version and does not say that the sources disagree.

{REPLY_FORMAT}"""

# Some models write their reasoning into the answer between <think> tags
THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)
VERDICT = re.compile(r"\W*(correct|partial|incorrect)\b\W*(.*)", re.IGNORECASE | re.DOTALL)


class Verdict(BaseModel):
    label: Label
    reason: str

    @property
    def score(self) -> float | None:
        # None for an unreadable judge reply: it is counted apart instead of as a wrong answer
        return SCORES.get(self.label)


def parse_verdict(text: str) -> Verdict:
    cleaned = normalize_answer(THINKING.sub("", text))
    match = VERDICT.match(cleaned)
    if not match:
        return Verdict(label="invalid", reason=cleaned[:300])
    return Verdict(label=match.group(1).lower(), reason=" ".join(match.group(2).split()))


class AnswerJudge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def grade(self, question: str, reference: str, answer: str, conflict: bool = False) -> Verdict:
        # Citation markers are not part of the answer the judge compares
        user = f"Question: {question}\nReference answer: {reference}\nSystem answer: {strip_citations(answer)}"
        system = CONFLICT_PROMPT if conflict else JUDGE_PROMPT
        messages: list[Message] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return parse_verdict(self.llm.chat(messages, max_tokens=300))


def grade_answer(
    judge: AnswerJudge, question: dict, answer: str, answered: bool, conflicts: Sequence[str] = ()
) -> Verdict:
    """Answer Correctness of one evaluation question; the judge is called only when it is needed.

    conflicts: the disagreements the system reported. The UI shows them next to the answer, so the
    judge grades both, as the user reads them.
    """
    if question["category"] == "unanswerable":
        if answered:
            return Verdict(label="incorrect", reason="Answered a question the documents cannot answer")
        return Verdict(label="correct", reason="Abstained: the documents do not hold the answer")
    if not answered:
        return Verdict(label="incorrect", reason="Abstained although the documents hold the answer")
    if conflicts:
        answer = f"{answer}\nSources disagree: {'; '.join(conflicts)}"
    conflict = question["category"] == "contradictory"
    return judge.grade(question["question"], question["answer"], answer, conflict=conflict)
