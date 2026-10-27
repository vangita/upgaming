import logging
import uuid
from functools import lru_cache

import numpy as np
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from app.core.config import get_settings
from app.ingestion.models import Chunk
from app.retrieval.bm25 import encode_document, encode_query

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"
INDEXED_FIELDS = ("source", "metadata.content_hash")


class SearchHit(BaseModel):
    id: str
    score: float
    chunk: Chunk


def chunk_id(chunk: Chunk) -> str:
    # Same document content + same position -> same ID,
    # so indexing a chunk twice overwrites it instead of duplicating it
    key = f"{chunk.metadata['content_hash']}/{chunk.chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _match(field: str, value: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=field, match=models.MatchValue(value=value))])


def _sparse(weights: dict[int, float]) -> models.SparseVector:
    return models.SparseVector(indices=list(weights), values=list(weights.values()))


@lru_cache
def get_client() -> QdrantClient:
    settings = get_settings()
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=60)


class QdrantStore:
    def __init__(self, collection: str, client: QdrantClient | None = None) -> None:
        self.collection = collection
        self.client = client or get_client()

    def ensure_collection(self, dimension: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return
        if exists:
            logger.info("Deleting collection %s", self.collection)
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dimension, distance=models.Distance.COSINE)},
            # Qdrant computes IDF over the whole collection; together with our
            # document weights the sparse dot product becomes the BM25 score
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field in INDEXED_FIELDS:
            self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)
        logger.info("Created collection %s (dimension %d)", self.collection, dimension)

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, batch_size: int = 256) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        # BM25 vectors are built from the text here, so the keyword index can never drift from it
        points = (
            models.PointStruct(
                id=chunk_id(chunk),
                vector={DENSE: vector.tolist(), SPARSE: _sparse(encode_document(chunk.text))},
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors)
        )
        self.client.upload_points(self.collection, points, batch_size=batch_size, wait=True)

    def semantic_search(
        self, query_vector: np.ndarray, limit: int = 10, source: str | None = None
    ) -> list[SearchHit]:
        return self._query(query_vector, DENSE, limit, source)

    def keyword_search(self, query: str, limit: int = 10, source: str | None = None) -> list[SearchHit]:
        weights = encode_query(query)
        if not weights:
            return []
        return self._query(_sparse(weights), SPARSE, limit, source)

    def _query(self, query, using: str, limit: int, source: str | None) -> list[SearchHit]:
        response = self.client.query_points(
            self.collection,
            query=query,
            using=using,
            limit=limit,
            query_filter=_match("source", source) if source else None,
            with_payload=True,
        )
        return [
            SearchHit(id=str(point.id), score=point.score, chunk=Chunk.model_validate(point.payload))
            for point in response.points
        ]

    def has_content(self, content_hash: str) -> bool:
        hash_filter = _match("metadata.content_hash", content_hash)
        return self.client.count(self.collection, count_filter=hash_filter).count > 0

    def delete_source(self, source: str) -> int:
        source_filter = _match("source", source)
        deleted = self.client.count(self.collection, count_filter=source_filter).count
        if deleted:
            self.client.delete(self.collection, points_selector=models.FilterSelector(filter=source_filter))
        logger.info("Deleted %d chunks of %s from %s", deleted, source, self.collection)
        return deleted

    def count(self) -> int:
        return self.client.count(self.collection).count
