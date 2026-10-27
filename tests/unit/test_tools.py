from app.ingestion.models import Chunk
from app.rag.tools import SEARCH_TOOL, SUBMIT_ANSWER_TOOL, SourceList, format_search_result
from app.storage.qdrant_store import SearchHit


def hit(title: str) -> SearchHit:
    text = f"Text about {title}."
    chunk = Chunk(text=text, title=title, source="test.md", file_format="md",
                  chunk_index=0, char_start=0, char_end=len(text))
    return SearchHit(id=title, score=0.5, chunk=chunk)


def test_numbers_continue_across_searches():
    sources = SourceList()
    sources.add([hit("A"), hit("B")])
    numbered = sources.add([hit("C"), hit("D")])
    assert [(n, h.id) for n, h, _ in numbered] == [(3, "C"), (4, "D")]
    assert [h.id for h in sources.hits] == ["A", "B", "C", "D"]


def test_a_source_found_again_keeps_its_number():
    sources = SourceList()
    sources.add([hit("A"), hit("B")])
    numbered = sources.add([hit("B"), hit("C")])
    assert [(n, h.id, new) for n, h, new in numbered] == [(2, "B", False), (3, "C", True)]
    assert len(sources.hits) == 3


def test_format_shows_text_only_for_new_sources():
    sources = SourceList()
    sources.add([hit("A")])
    text = format_search_result(sources.add([hit("A"), hit("B")]))
    assert "[1] A (already shown above)" in text
    assert "[2] B\nText about B." in text
    assert "Text about A." not in text


def test_format_empty_result():
    assert format_search_result([]) == "No sources found for this query."


def test_tool_schemas_have_the_expected_names_and_required_fields():
    assert SEARCH_TOOL["function"]["name"] == "search"
    assert SEARCH_TOOL["function"]["parameters"]["required"] == ["query"]
    assert SUBMIT_ANSWER_TOOL["function"]["parameters"]["required"] == ["answer", "sources", "answered", "conflicts"]
