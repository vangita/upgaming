import re

# [1], [1, 2] and the full-width 【1】 that gpt-oss models sometimes write
CITATION = re.compile(r"[\[【]\s*(\d+(?:\s*,\s*\d+)*)\s*[\]】]")

# gpt-oss writes look-alike characters that break exact text matching: narrow and regular
# no-break spaces (U+202F, U+00A0), the non-breaking hyphen (U+2011) and full-width brackets
LOOKALIKES = str.maketrans({" ": " ", "\xa0": " ", "‑": "-", "【": "[", "】": "]"})


def normalize_answer(text: str) -> str:
    return text.translate(LOOKALIKES).strip()


def extract_citations(text: str) -> list[int]:
    # Source numbers in order of first appearance, without duplicates
    numbers: list[int] = []
    for match in CITATION.finditer(text):
        for part in match.group(1).split(","):
            number = int(part)
            if number not in numbers:
                numbers.append(number)
    return numbers
