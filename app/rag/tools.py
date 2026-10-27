from app.storage.qdrant_store import SearchHit

SEARCH = "search"
SUBMIT_ANSWER = "submit_answer"

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": SEARCH,
        "description": (
            "Search the document collection for more sources. Use it when the sources you have "
            "do not contain a fact you need, e.g. a person or thing they only mention by name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A short, specific search query"},
            },
            "required": ["query"],
        },
    },
}

SUBMIT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": SUBMIT_ANSWER,
        "description": "Give the final answer. Call it exactly once, after any searches.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The answer with [n] citations"},
                "sources": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Numbers of all sources the answer relies on",
                },
                "answered": {
                    "type": "boolean",
                    "description": "false if the sources do not contain the answer",
                },
                "conflicts": {
                    "type": "array",
                    "description": "Disagreements between sources that matter for the question; empty if none",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string", "description": "What the sources disagree on"},
                            "sources": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Numbers of the disagreeing sources",
                            },
                        },
                        "required": ["description", "sources"],
                    },
                },
            },
            "required": ["answer", "sources", "answered", "conflicts"],
        },
    },
}


class SourceList:
    """Every source shown to the LLM during one question, numbered from 1 in order of discovery."""

    def __init__(self) -> None:
        self.hits: list[SearchHit] = []
        self._numbers: dict[str, int] = {}

    def add(self, hits: list[SearchHit]) -> list[tuple[int, SearchHit, bool]]:
        # (number, hit, is_new); a chunk found again keeps the number it got the first time
        numbered = []
        for hit in hits:
            is_new = hit.id not in self._numbers
            if is_new:
                self.hits.append(hit)
                self._numbers[hit.id] = len(self.hits)
            numbered.append((self._numbers[hit.id], hit, is_new))
        return numbered


def format_search_result(numbered: list[tuple[int, SearchHit, bool]]) -> str:
    if not numbered:
        return "No sources found for this query."
    return "\n\n".join(
        f"[{number}] {hit.chunk.title}\n{hit.chunk.text}" if is_new
        else f"[{number}] {hit.chunk.title} (already shown above)"
        for number, hit, is_new in numbered
    )
