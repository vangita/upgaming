import pytest

from app.ingestion.chunking.fixed import chunk_fixed
from app.ingestion.chunking.structured import chunk_structured
from app.ingestion.models import Document


def document(text: str) -> Document:
    return Document(title="Doc", text=text, source="doc.md", file_format="md")


def words(count: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


def sizes(chunks) -> list[int]:
    return [len(chunk.text.split()) for chunk in chunks]


def test_fixed_windows_overlap():
    chunks = chunk_fixed(document(words(300)), chunk_size=150, overlap=30)
    # Windows start every 120 words (150 - 30): at words 0, 120 and 240
    assert sizes(chunks) == [150, 150, 60]
    assert chunks[0].text.split()[-30:] == chunks[1].text.split()[:30]


def test_fixed_short_document_is_one_chunk():
    chunks = chunk_fixed(document("Only a few words here."))
    assert [chunk.text for chunk in chunks] == ["Only a few words here."]


def test_fixed_overlap_must_be_smaller_than_the_chunk():
    with pytest.raises(ValueError):
        chunk_fixed(document(words(10)), chunk_size=5, overlap=5)


def test_structured_keeps_each_section_with_its_heading():
    text = f"# Summary\n\n{words(40, 's')}\n\n## Root cause\n\n{words(40, 'r')}"
    chunks = chunk_structured(document(text))
    assert [chunk.text.split()[:2] for chunk in chunks] == [["#", "Summary"], ["##", "Root"]]


def test_structured_joins_a_short_section_to_the_next_one():
    # "# Intro" has 4 words, fewer than the 30 a chunk needs to stand alone
    chunks = chunk_structured(document(f"# Intro\n\nShort intro.\n\n## Details\n\n{words(50)}"))
    assert len(chunks) == 1 and chunks[0].text.startswith("# Intro")


def test_structured_splits_a_long_section_at_paragraphs():
    chunks = chunk_structured(document(f"# Report\n\n{words(100, 'a')}\n\n{words(100, 'b')}"))
    assert [chunk.text.split()[-1] for chunk in chunks] == ["a99", "b99"]
    assert chunks[1].text.split()[0] == "b0"


def test_structured_splits_a_long_paragraph_at_sentences():
    sentence = words(100) + "."
    assert sizes(chunk_structured(document(f"{sentence} {sentence}"))) == [100, 100]


def test_structured_cuts_text_without_sentences_into_words():
    assert sizes(chunk_structured(document(words(400)))) == [150, 150, 100]


def test_structured_joins_a_short_last_piece_to_the_chunk_before():
    # The last 10 words are too few for a chunk of their own, so the chunk before grows to 160
    assert sizes(chunk_structured(document(f"{words(150, 'a')}\n\n{words(10, 'b')}"))) == [160]


@pytest.mark.parametrize("chunker", [chunk_fixed, chunk_structured])
def test_chunks_point_into_the_document_text(chunker):
    doc = document(f"# Title\n\n{words(200, 'a')}.\n\n## Part two\n\n{words(120, 'b')}")
    chunks = chunker(doc)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert doc.text[chunk.char_start:chunk.char_end] == chunk.text
        assert (chunk.title, chunk.source) == ("Doc", "doc.md")
