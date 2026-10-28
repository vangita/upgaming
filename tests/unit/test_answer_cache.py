from app.api.cache import AnswerCache
from app.api.schemas import AskRequest, AskResponse


def request(question: str, method: str = "agent", search_mode: str = "hybrid") -> AskRequest:
    return AskRequest(question=question, method=method, search_mode=search_mode)


def response(answer: str) -> AskResponse:
    return AskResponse(
        question="q", answer=answer, answered=True, sources=[], citations=[], invalid_citations=[],
        conflicts=[], searches=[], duration_ms=10,
    )


def test_case_and_spaces_do_not_matter():
    cache = AnswerCache()
    cache.put(request("Who wrote the report?"), response("A"))
    assert cache.get(request("  who  wrote the REPORT? ")).answer == "A"


def test_method_and_search_mode_are_part_of_the_question():
    cache = AnswerCache()
    cache.put(request("Who?"), response("A"))
    assert cache.get(request("Who?", method="simple")) is None
    assert cache.get(request("Who?", search_mode="semantic")) is None


def test_the_least_recently_used_answer_goes_first():
    cache = AnswerCache(max_size=2)
    cache.put(request("one?"), response("1"))
    cache.put(request("two?"), response("2"))
    cache.get(request("one?"))  # "one" is now the most recent, so "two" goes
    cache.put(request("three?"), response("3"))
    assert cache.get(request("two?")) is None
    assert [cache.get(request(q)).answer for q in ("one?", "three?")] == ["1", "3"]
    assert len(cache) == 2


def test_size_zero_turns_the_cache_off():
    cache = AnswerCache(max_size=0)
    cache.put(request("Who?"), response("A"))
    assert cache.get(request("Who?")) is None


def test_clear():
    cache = AnswerCache()
    cache.put(request("Who?"), response("A"))
    cache.clear()
    assert cache.get(request("Who?")) is None
