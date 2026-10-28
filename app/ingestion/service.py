import logging
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from app.embeddings.embedder import Embedder, get_embedder
from app.ingestion.chunking.structured import chunk_structured
from app.ingestion.dedup import deduplicate
from app.ingestion.models import Chunk, Document
from app.ingestion.parsers.router import UnsupportedFormatError, parse_file
from app.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# The base dataset is indexed by scripts/index_chunks.py. Replacing it through an upload
# would delete ~70k chunks that take hours to embed again. The API can protect more files
# (settings.protected_sources); the command-line scripts manage every file
PROTECTED_SOURCES = frozenset({"hotpotqa_paragraphs.jsonl"})


class IngestionError(ValueError):
    """The file cannot be indexed; the message is safe to show to the user."""


class IngestResult(BaseModel):
    source: str
    # Documents indexed from the file (a PDF gives one per page) and their chunks
    documents: int
    chunks: int
    # Chunks of the previous version of the same file that were removed
    replaced: int
    # Titles of documents skipped because another file already has the same content
    duplicates: list[str]


class IngestionService:
    """Adds, updates and deletes files: parse -> dedup -> chunk -> embed -> upsert.

    A file is identified by its name: uploading the same name again replaces the old version.
    """

    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder | None = None,
        chunker: Callable[[Document], list[Chunk]] = chunk_structured,
        protected: frozenset[str] = PROTECTED_SOURCES,
    ) -> None:
        self.store = store
        self.embedder = embedder or get_embedder()
        self.chunker = chunker
        # Files that cannot be replaced or deleted through this service
        self.protected = protected

    def _check_not_protected(self, source: str) -> None:
        if source in self.protected:
            raise IngestionError(f"{source} is protected and cannot be changed through uploads")

    def ingest_file(self, path: Path) -> IngestResult:
        source = path.name
        self._check_not_protected(source)
        try:
            documents = deduplicate(parse_file(path))
        except UnsupportedFormatError as e:
            raise IngestionError(str(e)) from e
        if not documents:
            raise IngestionError(f"No text could be extracted from {source}")

        new, duplicates = [], []
        for document in documents:
            if self.store.has_content(document.metadata["content_hash"], exclude_source=source):
                duplicates.append(document.title)
            else:
                new.append(document)
        if duplicates:
            logger.info("%s: %d documents already indexed from other files", source, len(duplicates))

        chunks = [chunk for document in new for chunk in self.chunker(document)]
        # Embedding is the slow step that can fail, so it runs before the old version is deleted
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks]) if chunks else None
        replaced = self.store.delete_source(source)
        if chunks:
            self.store.upsert(chunks, vectors)

        logger.info("%s: %d documents, %d chunks indexed, %d old chunks replaced",
                    source, len(new), len(chunks), replaced)
        return IngestResult(
            source=source, documents=len(new), chunks=len(chunks), replaced=replaced, duplicates=duplicates
        )

    def delete(self, source: str) -> int:
        self._check_not_protected(source)
        return self.store.delete_source(source)
