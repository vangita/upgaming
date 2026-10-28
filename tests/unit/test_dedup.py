from app.ingestion.dedup import content_hash, deduplicate
from app.ingestion.models import Document


def document(title: str, text: str) -> Document:
    return Document(title=title, text=text, source="paragraphs.jsonl", file_format="jsonl")


def test_case_and_spacing_do_not_make_a_new_text():
    assert content_hash("The API  was\ndown.") == content_hash("the api was down.")


def test_any_other_change_does():
    assert content_hash("The API was down.") != content_hash("The API was down")


def test_duplicates_are_dropped_and_their_other_titles_kept_as_aliases():
    unique = deduplicate([
        document("A", "Same text."),
        document("B", "same  TEXT."),
        document("C", "Other text."),
        document("A", "Same text."),  # the same title again is not an alias
    ])
    assert [doc.title for doc in unique] == ["A", "C"]
    assert unique[0].metadata["aliases"] == ["B"]
    assert "aliases" not in unique[1].metadata


def test_kept_documents_carry_their_content_hash():
    unique = deduplicate([document("A", "Some text.")])
    assert unique[0].metadata["content_hash"] == content_hash("Some text.")
