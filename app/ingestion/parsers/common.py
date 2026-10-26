import re

FIRST_HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def first_heading(text: str) -> str | None:
    match = FIRST_HEADING.search(text)
    return match.group(1).strip() if match else None
