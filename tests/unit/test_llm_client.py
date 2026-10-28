import logging
from types import SimpleNamespace

import groq
import httpx
import pytest

from app.llm.client import DailyLimitError, LLMClient, LLMError, MinuteLimitError, Usage

MINUTE = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01abc` on tokens per "
          "minute (TPM): Limit 8000, Used 7500, Requested 1500. Please try again in 7.5s.")
DAY = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01abc` on tokens per "
       "day (TPD): Limit 200000, Used 199500, Requested 1500. Please try again in 9m0s.")


def client_with(create) -> LLMClient:
    # The SDK client is replaced by an object whose chat.completions.create is `create`
    client = LLMClient(api_key="test", model="test-model")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client


def rate_limited(message: str):
    def create(**kwargs):
        response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
        raise groq.RateLimitError(message, response=response, body=None)
    return create


def answer(prompt_tokens: int, completion_tokens: int):
    def create(**kwargs):
        message = SimpleNamespace(content="OK", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )
    return create


def test_per_minute_limit():
    with pytest.raises(MinuteLimitError):
        client_with(rate_limited(MINUTE)).chat([{"role": "user", "content": "Hi"}])


def test_daily_limit():
    with pytest.raises(DailyLimitError) as error:
        client_with(rate_limited(DAY)).chat([{"role": "user", "content": "Hi"}])
    # The API maps every LLMError to 503, so the daily limit needs no extra handling there
    assert isinstance(error.value, LLMError)


def test_rate_limit_is_logged_without_the_organization_id(caplog):
    with caplog.at_level(logging.WARNING), pytest.raises(DailyLimitError):
        client_with(rate_limited(DAY)).chat([{"role": "user", "content": "Hi"}])
    assert "tokens per day" in caplog.text
    assert "org_01abc" not in caplog.text


def test_usage_adds_up_over_calls():
    client = client_with(answer(prompt_tokens=100, completion_tokens=20))
    client.chat([{"role": "user", "content": "Hi"}])
    client.chat([{"role": "user", "content": "Hi again"}])
    assert client.usage == Usage(calls=2, prompt_tokens=200, completion_tokens=40)
