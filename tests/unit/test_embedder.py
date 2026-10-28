import numpy as np

from app.embeddings.embedder import Embedder


def fake_embedder() -> Embedder:
    # No model is loaded: each "vector" is the length of its text, so the order can be checked
    embedder = object.__new__(Embedder)
    embedder.dimension = 1
    embedder.query_instruction = "Query: "
    embedder.batches = []

    def embed(texts: list[str]) -> np.ndarray:
        embedder.batches.append(texts)
        return np.array([[len(text)] for text in texts], dtype=np.float32)

    embedder._embed = embed
    return embedder


def test_documents_come_back_in_the_original_order():
    vectors = fake_embedder().embed_documents(["aa", "a", "aaaa", "aaa", "aaaaa"], batch_size=2)
    assert vectors[:, 0].tolist() == [2, 1, 4, 3, 5]


def test_a_batch_holds_texts_of_similar_length():
    embedder = fake_embedder()
    embedder.embed_documents(["aa", "a", "aaaa", "aaa", "aaaaa"], batch_size=2)
    assert embedder.batches == [["aaaaa", "aaaa"], ["aaa", "aa"], ["a"]]


def test_no_documents():
    assert fake_embedder().embed_documents([]).shape == (0, 1)


def test_query_gets_the_instruction():
    embedder = fake_embedder()
    embedder.embed_query("Who?")
    assert embedder.batches == [["Query: Who?"]]
