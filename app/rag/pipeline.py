import logging
from collections.abc import Sequence

from pydantic import BaseModel

from app.llm.client import LLMClient, get_llm
from app.rag.citations import extract_citations, normalize_answer
from app.rag.prompts import NO_ANSWER, build_messages
from app.retrieval.retriever import Retriever, SearchMode
from app.storage.qdrant_store import SearchHit

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    number: int
    hit: SearchHit


class Conflict(BaseModel):
    description: str
    sources: list[int]


class RAGAnswer(BaseModel):
    question: str
    answer: str
    # False when the system abstained: the sources do not contain the answer
    answered: bool
    # Sources the answer cites, in order of first appearance
    citations: list[Citation]
    # Everything the LLM was shown, numbered from 1
    sources: list[SearchHit]
    # Cited numbers that match no source, e.g. [7] when there were 5 sources
    invalid_citations: list[int]
    # Disagreements between sources that the LLM reported
    conflicts: list[Conflict] = []
    # Follow-up search queries the agent ran, in order
    searches: list[str] = []

    @classmethod
    def build(
        cls,
        question: str,
        text: str,
        sources: list[SearchHit],
        answered: bool | None = None,
        conflicts: Sequence[Conflict] = (),
        searches: Sequence[str] = (),
        listed_sources: Sequence[int] = (),
    ) -> "RAGAnswer":
        # listed_sources: numbers the LLM returned separately; used when the text has no [n] markers.
        # An abstention has nothing to support, so its listed sources are ignored.
        text = normalize_answer(text) or NO_ANSWER
        # An empty or NO_ANSWER text is an abstention, whatever the LLM claimed in `answered`
        answered = answered is not False and text != NO_ANSWER

        def exists(number: int) -> bool:
            return 1 <= number <= len(sources)

        cited = extract_citations(text) or (list(dict.fromkeys(listed_sources)) if answered else [])
        citations = [Citation(number=n, hit=sources[n - 1]) for n in cited if exists(n)]
        invalid = [n for n in cited if not exists(n)]
        valid_conflicts = [
            conflict.model_copy(update={"sources": [n for n in conflict.sources if exists(n)]})
            for conflict in conflicts
            if any(exists(n) for n in conflict.sources)
        ]

        if invalid:
            logger.warning("Answer cites sources that do not exist: %s", invalid)
        if answered and not citations:
            logger.warning("Answer has no valid citations")
        return cls(
            question=question,
            answer=text,
            answered=answered,
            citations=citations,
            sources=sources,
            invalid_citations=invalid,
            conflicts=valid_conflicts,
            searches=list(searches),
        )


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient | None = None) -> None:
        self.retriever = retriever
        self.llm = llm or get_llm()

    def answer(self, question: str, mode: SearchMode = "hybrid", top_k: int = 10) -> RAGAnswer:
        sources = self.retriever.search(question, mode=mode, limit=top_k)
        if not sources:
            logger.info("No sources found, answering without calling the LLM")
            return RAGAnswer.build(question, NO_ANSWER, [])
        return RAGAnswer.build(question, self.llm.chat(build_messages(question, sources)), sources)
