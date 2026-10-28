import pytest

from app.evaluation.judge import CONFLICT_PROMPT, JUDGE_PROMPT, AnswerJudge, grade_answer, parse_verdict

ANSWERABLE = {"category": "multi_doc", "question": "Which airport is further north?", "answer": "Aberdeen"}
UNANSWERABLE = {"category": "unanswerable", "question": "When was Carice van Houten born?", "answer": None}
CONTRADICTORY = {
    "category": "contradictory", "question": "How long did INC-2041 last?",
    "answer": "Sources disagree: 47 minutes vs about 65 minutes",
}


class FakeLLM:
    def __init__(self, reply: str = "CORRECT\nSame airport.") -> None:
        self.reply = reply
        self.messages = []

    def chat(self, messages, max_tokens=1024):
        self.messages.append(messages)
        return self.reply


@pytest.mark.parametrize(
    ("reply", "label", "score"),
    [
        ("CORRECT\nSame airport.", "correct", 1.0),
        ("Partial\nOnly the date is right.", "partial", 0.5),
        ("INCORRECT\nA different airport.", "incorrect", 0.0),
        ("**INCORRECT** - a different airport.", "incorrect", 0.0),
    ],
)
def test_parse_verdict(reply, label, score):
    verdict = parse_verdict(reply)
    assert verdict.label == label
    assert verdict.score == score


def test_parse_verdict_keeps_the_reason():
    assert parse_verdict("INCORRECT\nA different\nairport.").reason == "A different airport."


def test_parse_verdict_skips_the_reasoning_of_thinking_models():
    assert parse_verdict("<think>Maybe CORRECT? No.</think>\nINCORRECT\nWrong.").label == "incorrect"


def test_correctness_is_not_read_as_correct():
    verdict = parse_verdict("Correctness cannot be judged.")
    assert verdict.label == "invalid"
    assert verdict.score is None


def test_judge_sees_question_reference_and_answer_without_citations():
    llm = FakeLLM()
    AnswerJudge(llm).grade("Which airport is further north?", "Aberdeen", "Aberdeen is further north [1][3].")
    prompt = llm.messages[0][1]["content"]
    assert "Which airport is further north?" in prompt
    assert "Reference answer: Aberdeen" in prompt
    assert "System answer: Aberdeen is further north." in prompt


def test_answered_question_goes_to_the_judge():
    llm = FakeLLM("INCORRECT\nA different airport.")
    verdict = grade_answer(AnswerJudge(llm), ANSWERABLE, "Valley International.", answered=True)
    assert verdict.label == "incorrect"
    assert len(llm.messages) == 1
    assert llm.messages[0][0]["content"] == JUDGE_PROMPT


def test_contradictory_question_is_graded_with_the_conflict_rules():
    llm = FakeLLM("INCORRECT\nOnly one version.")
    grade_answer(AnswerJudge(llm), CONTRADICTORY, "It lasted 47 minutes.", answered=True)
    assert llm.messages[0][0]["content"] == CONFLICT_PROMPT


def test_reported_conflicts_are_graded_with_the_answer():
    # The UI shows them in a box next to the answer: the user reads both
    llm = FakeLLM()
    conflicts = ["47 minutes in the post-mortem [1] vs about 65 minutes on the status page [2]"]
    grade_answer(AnswerJudge(llm), CONTRADICTORY, "It lasted 47 minutes [1].", answered=True, conflicts=conflicts)
    prompt = llm.messages[0][1]["content"]
    assert "System answer: It lasted 47 minutes.\nSources disagree: 47 minutes in the post-mortem" in prompt


def test_abstaining_on_an_answerable_question_is_incorrect_without_the_judge():
    llm = FakeLLM()
    assert grade_answer(AnswerJudge(llm), ANSWERABLE, "Not enough information.", answered=False).label == "incorrect"
    assert llm.messages == []


def test_abstaining_on_an_unanswerable_question_is_correct_without_the_judge():
    llm = FakeLLM()
    assert grade_answer(AnswerJudge(llm), UNANSWERABLE, "Not enough information.", answered=False).label == "correct"
    assert llm.messages == []


def test_answering_an_unanswerable_question_is_incorrect_without_the_judge():
    llm = FakeLLM()
    assert grade_answer(AnswerJudge(llm), UNANSWERABLE, "Born in 1976 [2].", answered=True).label == "incorrect"
    assert llm.messages == []
