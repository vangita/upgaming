import json

import numpy as np
import pytest

from app.ingestion.models import Chunk
from app.ingestion.service import IngestionError, IngestionService


class FakeStore:
    """Keeps chunks in a list and answers the same questions as QdrantStore."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def upsert(self, chunks, vectors):
        assert len(chunks) == len(vectors)
        self.chunks.extend(chunks)

    def delete_source(self, source):
        kept = [chunk for chunk in self.chunks if chunk.source != source]
        deleted = len(self.chunks) - len(kept)
        self.chunks = kept
        return deleted

    def has_content(self, content_hash, exclude_source=None):
        return any(
            chunk.metadata["content_hash"] == content_hash and chunk.source != exclude_source
            for chunk in self.chunks
        )


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return np.zeros((len(texts), 4))


@pytest.fixture
def service():
    return IngestionService(FakeStore(), embedder=FakeEmbedder())


def write(folder, name, text):
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_new_file_is_indexed(service, tmp_path):
    result = service.ingest_file(write(tmp_path, "report.md", "# Outage\n\nThe API was down for 40 minutes."))
    assert (result.source, result.documents, result.replaced, result.duplicates) == ("report.md", 1, 0, [])
    assert result.chunks == len(service.store.chunks) > 0
    assert service.store.chunks[0].title == "Outage"


def test_new_version_replaces_the_old_one(service, tmp_path):
    first = service.ingest_file(write(tmp_path, "report.md", "The API was down for 40 minutes."))
    second = service.ingest_file(write(tmp_path, "report.md", "The API was down for 55 minutes."))
    assert second.replaced == first.chunks
    assert [chunk.text for chunk in service.store.chunks] == ["The API was down for 55 minutes."]


def test_same_file_again_is_not_a_duplicate_of_itself(service, tmp_path):
    path = write(tmp_path, "report.md", "The API was down for 40 minutes.")
    service.ingest_file(path)
    result = service.ingest_file(path)
    assert result.duplicates == [] and result.documents == 1
    assert len(service.store.chunks) == result.chunks


def test_content_already_indexed_from_another_file_is_skipped(service, tmp_path):
    service.ingest_file(write(tmp_path, "a.txt", "The API was down for 40 minutes."))
    embedded = len(service.embedder.texts)
    result = service.ingest_file(write(tmp_path, "b.txt", "The  API was down\nfor 40 minutes. "))
    assert result.duplicates == ["b"] and result.documents == result.chunks == 0
    assert len(service.embedder.texts) == embedded
    assert {chunk.source for chunk in service.store.chunks} == {"a.txt"}


def test_duplicates_inside_one_file_are_indexed_once(service, tmp_path):
    lines = [json.dumps({"title": title, "text": "Same paragraph."}) for title in ("A", "B")]
    result = service.ingest_file(write(tmp_path, "notes.jsonl", "\n".join(lines)))
    assert result.documents == 1
    assert service.store.chunks[0].metadata["aliases"] == ["B"]


def test_delete_removes_every_chunk_of_the_file(service, tmp_path):
    service.ingest_file(write(tmp_path, "a.txt", "First file."))
    service.ingest_file(write(tmp_path, "b.txt", "Second file."))
    assert service.delete("a.txt") == 1
    assert {chunk.source for chunk in service.store.chunks} == {"b.txt"}


def test_base_dataset_cannot_be_replaced_or_deleted(service, tmp_path):
    with pytest.raises(IngestionError):
        service.ingest_file(write(tmp_path, "hotpotqa_paragraphs.jsonl", '{"title": "A", "text": "B"}'))
    with pytest.raises(IngestionError):
        service.delete("hotpotqa_paragraphs.jsonl")


def test_unsupported_format_is_an_ingestion_error(service, tmp_path):
    with pytest.raises(IngestionError, match="not supported"):
        service.ingest_file(write(tmp_path, "sheet.csv", "a,b\n1,2"))


def test_file_without_text_is_an_ingestion_error(service, tmp_path):
    with pytest.raises(IngestionError, match="No text"):
        service.ingest_file(write(tmp_path, "empty.txt", "   \n"))
    assert service.store.chunks == []
