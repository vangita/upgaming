import pytest

from app.ingestion.models import Chunk
from app.retrieval.fusion import reciprocal_rank_fusion
from app.storage.qdrant_store import SearchHit


def hit(hit_id: str, score: float = 0.5) -> SearchHit:
    chunk = Chunk(
        text=f"text of {hit_id}", title=hit_id, source="test.md", file_format="md",
        chunk_index=0, char_start=0, char_end=len(f"text of {hit_id}"),
    )
    return SearchHit(id=hit_id, score=score, chunk=chunk)


def ids(hits: list[SearchHit]) -> list[str]:
    return [h.id for h in hits]


def test_worked_example_order():
    semantic = [hit("A"), hit("B"), hit("C")]
    keyword = [hit("B"), hit("D"), hit("A")]
    assert ids(reciprocal_rank_fusion([semantic, keyword])) == ["B", "A", "D", "C"]


def test_worked_example_scores():
    semantic = [hit("A"), hit("B"), hit("C")]
    keyword = [hit("B"), hit("D"), hit("A")]
    scores = {h.id: h.score for h in reciprocal_rank_fusion([semantic, keyword])}
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["D"] == pytest.approx(1 / 62)
    assert scores["C"] == pytest.approx(1 / 63)


def test_original_scores_are_ignored():
    # BM25 scores (~20) and cosine scores (~0.8) are not comparable; only the rank matters
    semantic = [hit("A", score=0.9), hit("B", score=0.1)]
    keyword = [hit("B", score=25.0), hit("A", score=0.001)]
    scores = {h.id: h.score for h in reciprocal_rank_fusion([semantic, keyword])}
    assert scores["A"] == pytest.approx(scores["B"])


def test_custom_k():
    fused = reciprocal_rank_fusion([[hit("A"), hit("B")]], k=0)
    assert [h.score for h in fused] == pytest.approx([1 / 1, 1 / 2])


def test_single_list_keeps_its_order():
    hits = [hit("A"), hit("B"), hit("C")]
    assert ids(reciprocal_rank_fusion([hits])) == ["A", "B", "C"]


def test_one_empty_list():
    assert ids(reciprocal_rank_fusion([[], [hit("A"), hit("B")]])) == ["A", "B"]


def test_no_results_at_all():
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([]) == []


def test_limit():
    semantic = [hit(f"s{i}") for i in range(10)]
    keyword = [hit(f"k{i}") for i in range(10)]
    assert len(reciprocal_rank_fusion([semantic, keyword], limit=5)) == 5


def test_found_by_both_beats_found_by_one():
    semantic = [hit("only_semantic"), hit("both")]
    keyword = [hit("only_keyword"), hit("both")]
    assert reciprocal_rank_fusion([semantic, keyword])[0].id == "both"


def test_ties_keep_first_seen_order():
    # A and B both get 1/61; A was seen first, so it must stay first (reproducible results)
    assert ids(reciprocal_rank_fusion([[hit("A")], [hit("B")]])) == ["A", "B"]


def test_chunk_is_kept():
    original = hit("A")
    fused = reciprocal_rank_fusion([[original]])[0]
    assert fused.chunk == original.chunk


def test_input_hits_are_not_modified():
    original = hit("A", score=0.75)
    reciprocal_rank_fusion([[original], [original]])
    assert original.score == 0.75
