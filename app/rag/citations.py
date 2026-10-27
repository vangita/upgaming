import re

# [1], [1, 2] and the full-width 【1】 that gpt-oss models sometimes write
CITATION = re.compile(r"[\[【]\s*(\d+(?:\s*,\s*\d+)*)\s*[\]】]")


def normalize_answer(text: str) -> str:
    # gpt-oss writes narrow no-break spaces (U+202F) and full-width brackets
    return text.replace(" ", " ").replace("【", "[").replace("】", "]").strip()


def extract_citations(text: str) -> list[int]:
    # Source numbers in order of first appearance, without duplicates
    numbers: list[int] = []
    for match in CITATION.finditer(text):
        for part in match.group(1).split(","):
            number = int(part)
            if number not in numbers:
                numbers.append(number)
    return numbers
