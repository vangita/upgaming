import re
import unicodedata
import zlib
from collections import Counter

TOKEN = re.compile(r"\w+")
STOPWORDS = frozenset(
    "a an and are as at be but by for from had has have he her his i in is it its "
    "of on or s she t that the their them they this to was were what when where "
    "which who whom why with".split()
)

# BM25 parameters: K1 limits how much repeating a word helps,
# B controls how strongly long chunks are penalised
K1 = 1.2
B = 0.75
# Mean chunk length in tokens (after stopword removal) in the HotpotQA corpus
AVG_DOC_LENGTH = 55.0


def tokenize(text: str) -> list[str]:
    # NFKD splits "é" into "e" + accent, so "Beyoncé" and "Beyonce" match
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return [token for token in TOKEN.findall(text) if token not in STOPWORDS]


def term_id(token: str) -> int:
    # Python's hash() changes between runs; crc32 gives the same ID in every process
    return zlib.crc32(token.encode("utf-8"))


def encode_document(text: str) -> dict[int, float]:
    tokens = tokenize(text)
    length_norm = K1 * (1 - B + B * len(tokens) / AVG_DOC_LENGTH)
    return {
        term_id(token): tf * (K1 + 1) / (tf + length_norm)
        for token, tf in Counter(tokens).items()
    }


def encode_query(text: str) -> dict[int, float]:
    return {term_id(token): 1.0 for token in set(tokenize(text))}
