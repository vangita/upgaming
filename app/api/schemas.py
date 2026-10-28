from typing import Literal

from pydantic import BaseModel, Field

from app.rag.pipeline import Conflict, RAGAnswer
from app.retrieval.retriever import SearchMode

# agent: multi-step, the LLM can search again before answering
# simple: one search and one LLM call
Method = Literal["agent", "simple"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    method: Method = "agent"
    search_mode: SearchMode = "hybrid"


class Source(BaseModel):
    number: int
    title: str
    source: str
    file_format: str
    text: str
    page: int | None = None
    cited: bool


class AskResponse(BaseModel):
    question: str
    answer: str
    answered: bool
    # Every source the LLM saw, numbered as in the [n] markers of the answer
    sources: list[Source]
    # Cited source numbers in order of first appearance
    citations: list[int]
    invalid_citations: list[int]
    conflicts: list[Conflict]
    searches: list[str]
    duration_ms: int
    # True when the answer was given earlier to the same question and comes from the cache
    cached: bool = False

    @classmethod
    def from_answer(cls, answer: RAGAnswer, duration_ms: int) -> "AskResponse":
        cited = [citation.number for citation in answer.citations]
        sources = [
            Source(
                number=number,
                title=hit.chunk.title,
                source=hit.chunk.source,
                file_format=hit.chunk.file_format,
                text=hit.chunk.text,
                page=hit.chunk.metadata.get("page"),
                cited=number in cited,
            )
            for number, hit in enumerate(answer.sources, start=1)
        ]
        return cls(
            question=answer.question,
            answer=answer.answer,
            answered=answer.answered,
            sources=sources,
            citations=cited,
            invalid_citations=answer.invalid_citations,
            conflicts=answer.conflicts,
            searches=answer.searches,
            duration_ms=duration_ms,
        )


class DocumentInfo(BaseModel):
    source: str
    chunks: int
    # The base dataset (and on the demo, the sample documents) cannot be replaced or deleted
    protected: bool


class DeleteResponse(BaseModel):
    source: str
    deleted_chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    collection: str
    chunks: int
    llm_model: str
    embedding_model: str


class LatencyStats(BaseModel):
    # How many recent answers (not from the cache) the times are measured over
    answers: int
    mean_ms: int | None
    p50_ms: int | None
    p95_ms: int | None


class LLMStats(BaseModel):
    model: str
    fallback_model: str | None
    calls: int
    # Calls answered by the fallback model because the main model's daily limit was used up
    fallback_calls: int
    prompt_tokens: int
    completion_tokens: int


class StatsResponse(BaseModel):
    """What the server did since it started (GET /stats)."""

    uptime_s: int
    questions: int
    questions_by_method: dict[str, int]
    cache_hits: int
    # Answers in the cache right now
    cached_answers: int
    abstentions: int
    answers_with_conflicts: int
    latency: LatencyStats
    llm: LLMStats
    # Failed questions by cause: "llm" or "vector_db"
    errors: dict[str, int]
    # Files "uploaded" and "deleted" through the API
    documents: dict[str, int]
