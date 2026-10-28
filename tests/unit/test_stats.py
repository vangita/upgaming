from types import SimpleNamespace

import pytest

from app.api.schemas import AskResponse, Conflict
from app.api.stats import LATENCY_WINDOW, Stats, percentile
from app.llm.client import Usage


def answer(duration_ms: int = 100, cached: bool = False, answered: bool = True, conflict: bool = False) -> AskResponse:
    conflicts = [Conflict(description="47 vs 65 minutes", sources=[1, 2])] if conflict else []
    return AskResponse(
        question="q", answer="a", answered=answered, sources=[], citations=[], invalid_citations=[],
        conflicts=conflicts, searches=[], duration_ms=duration_ms, cached=cached,
    )


def llm(**usage) -> SimpleNamespace:
    return SimpleNamespace(model="main-model", fallback_model="small-model", usage=Usage(**usage))


@pytest.mark.parametrize(
    ("share", "expected"), [(0.5, 50), (0.95, 95), (0.01, 1), (1.0, 100)],
)
def test_percentile_of_1_to_100(share, expected):
    assert percentile(list(range(1, 101)), share) == expected


def test_percentile_of_few_values():
    assert percentile([200], 0.95) == 200
    assert percentile([100, 300], 0.5) == 100
    assert percentile([], 0.5) is None


def test_answers_are_counted_by_kind():
    stats = Stats()
    stats.record_answer("agent", answer(conflict=True))
    stats.record_answer("agent", answer(answered=False))
    stats.record_answer("simple", answer(cached=True))
    snapshot = stats.snapshot(llm(), cached_answers=2)
    assert snapshot.questions == 3
    assert snapshot.questions_by_method == {"agent": 2, "simple": 1}
    assert (snapshot.cache_hits, snapshot.abstentions, snapshot.answers_with_conflicts) == (1, 1, 1)
    assert snapshot.cached_answers == 2


def test_cached_answers_are_left_out_of_the_answer_times():
    stats = Stats()
    stats.record_answer("agent", answer(duration_ms=1000))
    stats.record_answer("agent", answer(duration_ms=3000))
    stats.record_answer("agent", answer(duration_ms=0, cached=True))
    latency = stats.snapshot(llm(), 0).latency
    assert (latency.answers, latency.mean_ms, latency.p50_ms, latency.p95_ms) == (2, 2000, 1000, 3000)


def test_answer_times_cover_only_the_most_recent_answers():
    stats = Stats()
    for duration in range(LATENCY_WINDOW + 10):
        stats.record_answer("agent", answer(duration_ms=duration))
    latency = stats.snapshot(llm(), 0).latency
    assert latency.answers == LATENCY_WINDOW
    assert latency.p50_ms > 10  # the 10 oldest (and fastest) answers are gone


def test_no_answers_yet():
    latency = Stats().snapshot(llm(), 0).latency
    assert (latency.answers, latency.mean_ms, latency.p50_ms) == (0, None, None)


def test_llm_usage_errors_and_documents():
    stats = Stats()
    stats.record_error("llm")
    stats.record_error("llm")
    stats.record_document("uploaded")
    snapshot = stats.snapshot(llm(calls=5, fallback_calls=2, prompt_tokens=900, completion_tokens=100), 0)
    assert snapshot.errors == {"llm": 2}
    assert snapshot.documents == {"uploaded": 1}
    assert (snapshot.llm.calls, snapshot.llm.fallback_calls, snapshot.llm.prompt_tokens) == (5, 2, 900)
    assert snapshot.llm.fallback_model == "small-model"
