"""Evaluation metrics.

Retrieval metrics count documents, not chunks: gold documents are titles, and the retriever can
return several chunks of one document. `ranked_documents` turns the chunk ranking into a document
ranking first.
"""
from app.storage.qdrant_store import SearchHit


def document_title(hit: SearchHit, gold: set[str]) -> str:
    """The title a hit counts as. A paragraph that appeared under several titles is kept once by the
    deduplication, with the other titles in metadata["aliases"]: a gold alias counts as found."""
    titles = [hit.chunk.title, *hit.chunk.metadata.get("aliases", [])]
    return next((title for title in titles if title in gold), hit.chunk.title)


def ranked_documents(titles: list[str]) -> list[str]:
    """Each document once, at the rank of its best chunk: ["A", "B", "A", "C"] -> ["A", "B", "C"]."""
    return list(dict.fromkeys(titles))


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Share of the gold documents that are among the first k ranked documents.

    ranked: document titles, best first, each title once (see ranked_documents).
    gold: the question's gold documents. If it is empty, raise ValueError: recall has no meaning
    without gold documents (unanswerable questions are left out of the retrieval metrics).

    Example: ranked = ["Report", "Post-mortem", "Support", "Status page"],
             gold = {"Post-mortem", "Support"}  ->  k=1: 0.0, k=2: 0.5, k=3: 1.0
    """
    if not gold:
        raise ValueError("Recall@K needs at least one gold document")
    found = [title for title in ranked[:k] if title in gold]
    return len(found) / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    """1 / position of the first gold document in the ranking, counting positions from 1;
    0.0 if no gold document is in the ranking. MRR is the mean of this value over all questions.

    Example: ranked = ["Report", "Post-mortem", "Support", "Status page"],
             gold = {"Post-mortem", "Support"}  ->  0.5
    """
    for position, title in enumerate(ranked, start=1):
        if title in gold:
            return 1 / position
    return 0.0


def citation_accuracy(cited: list[str], gold: set[str]) -> float | None:
    """Share of the citations that point to a gold document; None when the answer cites nothing.

    cited: the document title of every cited source, so two cited chunks of one document count twice.
    Strict: citing another document that happens to state the same fact counts as a wrong citation.
    """
    if not cited:
        return None
    return sum(title in gold for title in cited) / len(cited)


def citation_coverage(cited: list[str], gold: set[str]) -> float | None:
    """Share of the gold documents cited at least once; None for questions without gold documents.

    A multi-hop answer has to cite every link of the chain: citing one of two gold documents gives 0.5.
    """
    if not gold:
        return None
    return len(gold & set(cited)) / len(gold)


def mean(values: list[float | None]) -> float | None:
    """Mean of the defined values. None marks a metric that does not apply to a question."""
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None
