from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import ResponseHandlingException

from app.api import main
from app.api.dependencies import get_agent, get_ingestion, get_pipeline, get_store
from app.ingestion.models import Chunk
from app.ingestion.service import PROTECTED_SOURCES, IngestionService
from app.llm.client import LLMError
from app.rag.pipeline import RAGAnswer
from app.storage.qdrant_store import SearchHit


def hit(title: str) -> SearchHit:
    text = f"Text about {title}."
    chunk = Chunk(text=text, title=title, source=f"{title}.md", file_format="md",
                  chunk_index=0, char_start=0, char_end=len(text))
    return SearchHit(id=title, score=0.5, chunk=chunk)


class FakeStore:
    """In-memory store with the methods the API and IngestionService use."""

    collection = "test_collection"

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.down = False

    def count(self):
        if self.down:
            raise ResponseHandlingException(ConnectionError("connection refused"))
        return len(self.chunks)

    def list_sources(self):
        return dict(Counter(chunk.source for chunk in self.chunks))

    def upsert(self, chunks, vectors):
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
    def embed_documents(self, texts):
        return np.zeros((len(texts), 4))


class FakeRAG:
    """Stands in for InvestigationAgent and RAGPipeline: both have answer(question, mode)."""

    def __init__(self, text: str = "Fact [2].", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def answer(self, question, mode="hybrid"):
        self.calls.append((question, mode))
        if self.error:
            raise self.error
        return RAGAnswer.build(question, self.text, [hit("A"), hit("B")])


@pytest.fixture
def api():
    # The lambdas read `state` on every request, so a test can swap a fake before calling
    state = SimpleNamespace(
        store=FakeStore(), agent=FakeRAG(), pipeline=FakeRAG("Simple [1]."), protected=PROTECTED_SOURCES,
    )
    main.app.dependency_overrides.update({
        get_store: lambda: state.store,
        get_agent: lambda: state.agent,
        get_pipeline: lambda: state.pipeline,
        get_ingestion: lambda: IngestionService(state.store, embedder=FakeEmbedder(), protected=state.protected),
    })
    # Without `with`, TestClient skips the lifespan, so no embedding model is loaded
    state.client = TestClient(main.app)
    yield state
    main.app.dependency_overrides.clear()


def upload(api, name, content):
    return api.client.post("/documents", files={"file": (name, content)})


def test_health_reports_the_collection(api):
    response = api.client.get("/health")
    assert response.status_code == 200
    assert response.json()["collection"] == "test_collection" and response.json()["chunks"] == 0


def test_ask_uses_the_agent_by_default(api):
    response = api.client.post("/ask", json={"question": "Who is B?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Fact [2]." and body["citations"] == [2]
    assert [(s["number"], s["title"], s["cited"]) for s in body["sources"]] == [(1, "A", False), (2, "B", True)]
    assert api.agent.calls == [("Who is B?", "hybrid")] and api.pipeline.calls == []


def test_ask_can_use_the_simple_pipeline_and_another_search_mode(api):
    body = {"question": "Who is A?", "method": "simple", "search_mode": "keyword"}
    response = api.client.post("/ask", json=body)
    assert response.json()["answer"] == "Simple [1]."
    assert api.pipeline.calls == [("Who is A?", "keyword")] and api.agent.calls == []


def test_ask_validates_the_request(api):
    response = api.client.post("/ask", json={"question": "x", "method": "magic"})
    assert response.status_code == 422
    assert {error["loc"][-1] for error in response.json()["detail"]} == {"question", "method"}


def test_llm_error_becomes_503_with_its_message(api):
    api.agent = FakeRAG(error=LLMError("LLM rate limit reached, please try again in a minute"))
    response = api.client.post("/ask", json={"question": "Who is B?"})
    assert response.status_code == 503
    assert response.json() == {"detail": "LLM rate limit reached, please try again in a minute"}


def test_vector_database_error_becomes_503(api):
    api.store.down = True
    response = api.client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"detail": "The vector database is unavailable"}


def test_upload_indexes_the_file_and_lists_it(api):
    response = upload(api, "report.md", b"# Outage\n\nThe API was down for 40 minutes.")
    assert response.status_code == 201
    assert response.json()["source"] == "report.md" and response.json()["chunks"] == 1
    listed = api.client.get("/documents").json()
    assert listed == [{"source": "report.md", "chunks": 1, "protected": False}]


def test_upload_keeps_only_the_file_name(api):
    response = upload(api, "..\\..\\app/../evil.md", b"Some text.")
    assert response.json()["source"] == "evil.md"


def test_upload_without_a_name_is_rejected(api):
    assert upload(api, "..", b"Some text.").status_code == 400


def test_unsupported_format_is_400(api):
    response = upload(api, "sheet.csv", b"a,b\n1,2")
    assert response.status_code == 400 and "not supported" in response.json()["detail"]


def test_too_large_upload_is_413(api, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    assert upload(api, "report.md", b"x" * 11).status_code == 413
    assert upload(api, "report.md", b"x" * 10).status_code == 201


def test_delete_removes_the_file(api):
    upload(api, "report.md", b"The API was down for 40 minutes.")
    response = api.client.delete("/documents/report.md")
    assert response.json() == {"source": "report.md", "deleted_chunks": 1}
    assert api.client.delete("/documents/report.md").status_code == 404


def test_base_dataset_cannot_be_deleted(api):
    response = api.client.delete("/documents/hotpotqa_paragraphs.jsonl")
    assert response.status_code == 400 and "protected" in response.json()["detail"]


def test_files_protected_by_the_deployment_cannot_be_changed(api):
    upload(api, "sample.md", b"The API was down for 40 minutes.")
    api.protected = PROTECTED_SOURCES | {"sample.md"}
    assert api.client.get("/documents").json() == [{"source": "sample.md", "chunks": 1, "protected": True}]
    assert api.client.delete("/documents/sample.md").status_code == 400
    assert upload(api, "sample.md", b"A new version.").status_code == 400
    assert api.client.get("/documents").json()[0]["chunks"] == 1
