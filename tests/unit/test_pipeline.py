from app.ingestion.models import Chunk
from app.rag.pipeline import RAGPipeline
from app.rag.prompts import NO_ANSWER
from app.storage.qdrant_store import SearchHit


def hit(title: str) -> SearchHit:
    text = f"Text about {title}."
    chunk = Chunk(text=text, title=title, source="test.md", file_format="md",
                  chunk_index=0, char_start=0, char_end=len(text))
    return SearchHit(id=title, score=0.5, chunk=chunk)


class FakeRetriever:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits

    def search(self, query, mode="hybrid", limit=10, source=None):
        return self.hits[:limit]


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self.reply


def make_pipeline(hits, reply):
    llm = FakeLLM(reply)
    return RAGPipeline(FakeRetriever(hits), llm=llm), llm


def test_citations_point_to_the_right_sources():
    pipeline, _ = make_pipeline([hit("A"), hit("B"), hit("C")], "Fact one [3]. Fact two [1].")
    result = pipeline.answer("question?")
    assert [(c.number, c.hit.chunk.title) for c in result.citations] == [(3, "C"), (1, "A")]
    assert result.invalid_citations == []


def test_citation_numbers_without_a_source_are_reported():
    pipeline, _ = make_pipeline([hit("A"), hit("B")], "Fact [2]. Made up [7]. Zero [0].")
    result = pipeline.answer("question?")
    assert [c.number for c in result.citations] == [2]
    assert result.invalid_citations == [7, 0]


def test_answer_is_normalized():
    pipeline, _ = make_pipeline([hit("A")], "Tim Burton【1】.")
    result = pipeline.answer("question?")
    assert result.answer == "Tim Burton[1]."
    assert [c.number for c in result.citations] == [1]


def test_no_sources_means_no_llm_call():
    pipeline, llm = make_pipeline([], "should never be used")
    result = pipeline.answer("question?")
    assert result.answer == NO_ANSWER
    assert result.sources == [] and result.citations == []
    assert llm.calls == []


def test_llm_sees_the_question_and_numbered_sources():
    pipeline, llm = make_pipeline([hit("A"), hit("B")], NO_ANSWER)
    pipeline.answer("Who is B?")
    user_message = llm.calls[0][1]["content"]
    assert "[1] A" in user_message and "[2] B" in user_message
    assert user_message.endswith("Question: Who is B?")


def test_top_k_limits_the_sources():
    pipeline, _ = make_pipeline([hit(t) for t in "ABCDEF"], NO_ANSWER)
    assert len(pipeline.answer("question?", top_k=3).sources) == 3
