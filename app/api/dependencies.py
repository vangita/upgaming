from functools import lru_cache

from app.api.cache import AnswerCache
from app.api.stats import Stats
from app.core.config import get_settings
from app.ingestion.service import PROTECTED_SOURCES, IngestionService
from app.rag.agent import InvestigationAgent
from app.rag.pipeline import RAGPipeline
from app.retrieval.retriever import Retriever
from app.storage.qdrant_store import QdrantStore

# One instance of each per process: they hold the embedding model and the HTTP clients.
# Routes get them through Depends(...), so tests can replace them with app.dependency_overrides


@lru_cache
def get_store() -> QdrantStore:
    return QdrantStore(get_settings().main_collection)


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(get_store())


@lru_cache
def get_agent() -> InvestigationAgent:
    return InvestigationAgent(get_retriever())


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(get_retriever())


@lru_cache
def get_stats() -> Stats:
    return Stats()


@lru_cache
def get_cache() -> AnswerCache:
    return AnswerCache(get_settings().answer_cache_size)


@lru_cache
def get_ingestion() -> IngestionService:
    # Shares the retriever's embedder, so the model is loaded only once. On the public demo the
    # sample documents are protected too, so a visitor cannot delete them for everyone
    protected = PROTECTED_SOURCES | frozenset(get_settings().protected_sources)
    return IngestionService(get_store(), get_retriever().embedder, protected=protected)
