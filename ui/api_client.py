import os
from typing import Any
from urllib.parse import quote

import httpx

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")


class APIError(RuntimeError):
    """The backend could not be reached or rejected the request; the message is shown in the UI."""


def _detail(response: httpx.Response) -> str:
    # FastAPI errors look like {"detail": "..."}; validation errors carry a list of problems
    try:
        body = response.json()
    except ValueError:
        body = None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list):
        detail = "; ".join(f"{problem['loc'][-1]}: {problem['msg']}" for problem in detail)
    return f"{detail or response.reason_phrase} (HTTP {response.status_code})"


class APIClient:
    """Thin HTTP client for the FastAPI backend. The UI imports nothing else from the backend."""

    def __init__(self, base_url: str = API_URL, transport: httpx.BaseTransport | None = None) -> None:
        # The agent may call the LLM several times and wait for rate limits, so reads get a long timeout
        timeout = httpx.Timeout(10.0, read=180.0)
        self.client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.TimeoutException as e:
            raise APIError("The backend did not answer in time") from e
        except httpx.TransportError as e:
            raise APIError(f"Cannot reach the backend at {self.client.base_url}") from e
        if response.is_error:
            raise APIError(_detail(response))
        return response.json()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def ask(self, question: str, method: str = "agent", search_mode: str = "hybrid") -> dict:
        body = {"question": question, "method": method, "search_mode": search_mode}
        return self._request("POST", "/ask", json=body)

    def documents(self) -> list[dict]:
        return self._request("GET", "/documents")

    def upload(self, name: str, content: bytes) -> dict:
        return self._request("POST", "/documents", files={"file": (name, content)})

    def delete(self, source: str) -> dict:
        # quote: file names can contain spaces, "#" or "?"
        return self._request("DELETE", f"/documents/{quote(source, safe='')}")
