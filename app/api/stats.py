import math
import time
from collections import Counter, deque
from threading import Lock

from app.api.schemas import AskResponse, LatencyStats, LLMStats, StatsResponse
from app.llm.client import LLMClient

# The answer times kept for the latency numbers: the most recent ones
LATENCY_WINDOW = 500


def percentile(sorted_values: list[int], share: float) -> int | None:
    """Nearest-rank percentile: the smallest value that `share` of the values are at or below."""
    if not sorted_values:
        return None
    rank = max(1, math.ceil(share * len(sorted_values)))
    return sorted_values[rank - 1]


class Stats:
    """What the server did since it started, for GET /stats. Kept in memory: a restart resets it."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        # Routes run in a thread pool: two requests can count at the same time
        self._lock = Lock()
        self.questions: Counter[str] = Counter()
        self.cache_hits = 0
        self.abstentions = 0
        self.answers_with_conflicts = 0
        self.errors: Counter[str] = Counter()
        self.documents: Counter[str] = Counter()
        self._durations: deque[int] = deque(maxlen=LATENCY_WINDOW)

    def record_answer(self, method: str, answer: AskResponse) -> None:
        with self._lock:
            self.questions[method] += 1
            if answer.cached:
                self.cache_hits += 1
            else:
                # A cached answer takes ~0 ms and would hide how long answering really takes
                self._durations.append(answer.duration_ms)
            self.abstentions += not answer.answered
            self.answers_with_conflicts += bool(answer.conflicts)

    def record_error(self, kind: str) -> None:
        with self._lock:
            self.errors[kind] += 1

    def record_document(self, action: str) -> None:
        with self._lock:
            self.documents[action] += 1

    def snapshot(self, llm: LLMClient, cached_answers: int) -> StatsResponse:
        with self._lock:
            durations = sorted(self._durations)
            return StatsResponse(
                uptime_s=round(time.monotonic() - self.started),
                questions=sum(self.questions.values()),
                questions_by_method=dict(self.questions),
                cache_hits=self.cache_hits,
                cached_answers=cached_answers,
                abstentions=self.abstentions,
                answers_with_conflicts=self.answers_with_conflicts,
                latency=LatencyStats(
                    answers=len(durations),
                    mean_ms=round(sum(durations) / len(durations)) if durations else None,
                    p50_ms=percentile(durations, 0.50),
                    p95_ms=percentile(durations, 0.95),
                ),
                llm=LLMStats(
                    model=llm.model,
                    fallback_model=llm.fallback_model,
                    calls=llm.usage.calls,
                    fallback_calls=llm.usage.fallback_calls,
                    prompt_tokens=llm.usage.prompt_tokens,
                    completion_tokens=llm.usage.completion_tokens,
                ),
                errors=dict(self.errors),
                documents=dict(self.documents),
            )
