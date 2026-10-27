import logging

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


class RAGAnswer(BaseModel):
    question: str
    answer: str
    # Sources the answer cites, in order of first appearance
    citations: list[Citation]
    # Everything the LLM was shown, numbered from 1
    sources: list[SearchHit]
    # Cited numbers that match no source, e.g. [7] when there were 5 sources
    invalid_citations: list[int]


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient | None = None) -> None:
        self.retriever = retriever
        self.llm = llm or get_llm()

    def answer(self, question: str, mode: SearchMode = "hybrid", top_k: int = 10) -> RAGAnswer:
        sources = self.retriever.search(question, mode=mode, limit=top_k)
        if not sources:
            logger.info("No sources found, answering without calling the LLM")
            return RAGAnswer(question=question, answer=NO_ANSWER, citations=[], sources=[], invalid_citations=[])

        text = normalize_answer(self.llm.chat(build_messages(question, sources)))
        cited = extract_citations(text)
        citations = [Citation(number=n, hit=sources[n - 1]) for n in cited if 1 <= n <= len(sources)]
        invalid = [n for n in cited if not 1 <= n <= len(sources)]

        if invalid:
            logger.warning("Answer cites sources that do not exist: %s", invalid)
        if not cited and text != NO_ANSWER:
            logger.warning("Answer has no citations")
        return RAGAnswer(
            question=question, answer=text, citations=citations, sources=sources, invalid_citations=invalid
        )
