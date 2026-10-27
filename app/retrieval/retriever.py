from typing import Literal

from app.embeddings.embedder import Embedder, get_embedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.storage.qdrant_store import QdrantStore, SearchHit

SearchMode = Literal["semantic", "keyword", "hybrid"]


class Retriever:
    def __init__(self, store: QdrantStore, embedder: Embedder | None = None, candidates: int = 30) -> None:
        self.store = store
        self.embedder = embedder or get_embedder()
        # How many hits each method contributes to the fusion in hybrid mode
        self.candidates = candidates

    def search(
        self, query: str, mode: SearchMode = "hybrid", limit: int = 10, source: str | None = None
    ) -> list[SearchHit]:
        if mode == "semantic":
            return self._semantic(query, limit, source)
        if mode == "keyword":
            return self.store.keyword_search(query, limit, source)
        if mode == "hybrid":
            depth = max(limit, self.candidates)
            semantic = self._semantic(query, depth, source)
            keyword = self.store.keyword_search(query, depth, source)
            return reciprocal_rank_fusion([semantic, keyword], limit=limit)
        raise ValueError(f"Unknown search mode: {mode}")

    def _semantic(self, query: str, limit: int, source: str | None) -> list[SearchHit]:
        return self.store.semantic_search(self.embedder.embed_query(query), limit, source)
