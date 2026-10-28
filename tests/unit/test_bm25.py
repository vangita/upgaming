import pytest

from app.retrieval.bm25 import AVG_DOC_LENGTH, B, K1, encode_document, encode_query, term_id, tokenize


def weight(text: str, word: str) -> float:
    return encode_document(text)[term_id(word)]


def test_tokenize_lowercases_folds_accents_and_drops_stopwords():
    assert tokenize("The Beyoncé concert in Paris") == ["beyonce", "concert", "paris"]


def test_identifiers_are_split_into_their_parts():
    assert tokenize("Error GW-503 on INC-2041") == ["error", "gw", "503", "inc", "2041"]


def test_document_weight_is_the_bm25_term_frequency_part():
    # 4 tokens, "outage" twice; the IDF part is added by Qdrant (Modifier.IDF)
    length_norm = K1 * (1 - B + B * 4 / AVG_DOC_LENGTH)
    assert weight("outage outage payment card", "outage") == pytest.approx(2 * (K1 + 1) / (2 + length_norm))


def test_repeating_a_word_helps_less_and_less():
    # Same length (20 tokens), the word 1, 2, 4 and 8 times
    w1, w2, w4, w8 = (weight(" ".join(["outage"] * n + ["filler"] * (20 - n)), "outage") for n in (1, 2, 4, 8))
    assert w1 < w2 < w4 < w8 < K1 + 1
    # Gain per extra occurrence keeps falling
    assert (w2 - w1) > (w4 - w2) / 2 > (w8 - w4) / 4


def test_a_longer_chunk_gets_a_lower_weight_for_the_same_count():
    short = "outage " + " ".join(f"f{i}" for i in range(5))
    long = "outage " + " ".join(f"f{i}" for i in range(100))
    assert weight(long, "outage") < weight(short, "outage")


def test_query_has_weight_one_per_distinct_word():
    assert encode_query("Outage OUTAGE payment") == {term_id("outage"): 1.0, term_id("payment"): 1.0}


def test_query_and_documents_use_the_same_term_ids():
    assert set(encode_query("payment outage")) <= set(encode_document("The payment outage lasted an hour"))
