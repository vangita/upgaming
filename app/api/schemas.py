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
    # The base dataset cannot be replaced or deleted through the API
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
