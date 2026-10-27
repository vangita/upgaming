import httpx
import pytest

from ui.api_client import APIClient, APIError


def client_with(handler) -> APIClient:
    # MockTransport answers requests with `handler` instead of sending them over the network
    return APIClient("http://backend", transport=httpx.MockTransport(handler))


def test_ask_sends_the_question_and_settings():
    seen = {}

    def handler(request):
        seen["path"], seen["body"] = request.url.path, request.read()
        return httpx.Response(200, json={"answer": "Fact [1]."})

    assert client_with(handler).ask("Who?", "simple", "keyword") == {"answer": "Fact [1]."}
    assert seen["path"] == "/ask"
    assert seen["body"] == b'{"question":"Who?","method":"simple","search_mode":"keyword"}'


def test_error_detail_from_the_backend_is_shown():
    client = client_with(lambda request: httpx.Response(503, json={"detail": "LLM rate limit reached"}))
    with pytest.raises(APIError, match=r"LLM rate limit reached \(HTTP 503\)"):
        client.health()


def test_validation_errors_are_joined():
    detail = [{"loc": ["body", "question"], "msg": "too short"}, {"loc": ["body", "method"], "msg": "invalid"}]
    client = client_with(lambda request: httpx.Response(422, json={"detail": detail}))
    with pytest.raises(APIError, match="question: too short; method: invalid"):
        client.ask("x")


def test_unreachable_backend():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(APIError, match="Cannot reach the backend"):
        client_with(handler).documents()


def test_file_names_are_quoted_in_the_url():
    paths = []

    def handler(request):
        paths.append(request.url.raw_path)
        return httpx.Response(200, json={"source": "my report #2.md", "deleted_chunks": 1})

    client_with(handler).delete("my report #2.md")
    assert paths == [b"/documents/my%20report%20%232.md"]
