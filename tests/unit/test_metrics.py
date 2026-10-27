import pytest

from app.evaluation.metrics import (
    citation_accuracy,
    citation_coverage,
    mean,
    ranked_documents,
    recall_at_k,
    reciprocal_rank,
)

# Question inc-mh1 from stage 11.1: six chunks from four documents
CHUNK_TITLES = ["Report", "Post-mortem", "Support", "Post-mortem", "Status page", "Report"]
RANKED = ["Report", "Post-mortem", "Support", "Status page"]
GOLD = {"Post-mortem", "Support"}


def test_ranked_documents_keeps_each_document_at_its_best_rank():
    assert ranked_documents(CHUNK_TITLES) == RANKED


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.5), (3, 1.0), (4, 1.0)])
def test_recall_at_k_worked_example(k, expected):
    assert recall_at_k(RANKED, GOLD, k) == pytest.approx(expected)


def test_recall_divides_by_the_number_of_gold_documents_not_by_k():
    # One of two gold documents in the top 4: 1/2, not 1/4
    assert recall_at_k(["A", "B", "C", "D"], {"B", "Z"}, 4) == pytest.approx(0.5)


def test_recall_when_k_is_larger_than_the_ranking():
    assert recall_at_k(["A", "B"], {"B"}, 10) == pytest.approx(1.0)


def test_recall_when_no_gold_document_is_found():
    assert recall_at_k(RANKED, {"Z"}, 3) == 0.0


def test_recall_needs_gold_documents():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 3)


def test_reciprocal_rank_worked_example():
    assert reciprocal_rank(RANKED, GOLD) == pytest.approx(0.5)


def test_reciprocal_rank_of_a_gold_document_at_the_top():
    assert reciprocal_rank(["Support", "Report"], GOLD) == pytest.approx(1.0)


def test_reciprocal_rank_counts_positions_from_one():
    assert reciprocal_rank(["A", "B", "G"], {"G"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_only_the_first_gold_document():
    # Gold documents at positions 2 and 4: 1/2, not 1/4 and not their mean
    assert reciprocal_rank(["A", "G1", "B", "G2"], {"G1", "G2"}) == pytest.approx(0.5)


def test_reciprocal_rank_without_a_gold_document_in_the_ranking():
    assert reciprocal_rank(RANKED, {"Z"}) == 0.0


def test_reciprocal_rank_of_an_empty_ranking():
    assert reciprocal_rank([], GOLD) == 0.0


def test_mrr_is_the_mean_of_the_reciprocal_ranks():
    rankings = [(["G"], {"G"}), (["A", "G"], {"G"}), (["A"], {"G"})]
    assert mean([reciprocal_rank(r, g) for r, g in rankings]) == pytest.approx((1 + 0.5 + 0) / 3)


def test_citation_accuracy_counts_every_citation():
    assert citation_accuracy(["Post-mortem", "Post-mortem", "Report"], GOLD) == pytest.approx(2 / 3)


def test_citation_accuracy_without_citations_is_undefined():
    assert citation_accuracy([], GOLD) is None


def test_citation_coverage_needs_every_gold_document():
    assert citation_coverage(["Post-mortem", "Post-mortem"], GOLD) == pytest.approx(0.5)
    assert citation_coverage(["Support", "Post-mortem", "Report"], GOLD) == pytest.approx(1.0)


def test_citation_coverage_without_gold_documents_is_undefined():
    assert citation_coverage(["Report"], set()) is None


def test_mean_skips_undefined_values():
    assert mean([1.0, None, 0.5]) == pytest.approx(0.75)
    assert mean([None]) is None
    assert mean([]) is None
