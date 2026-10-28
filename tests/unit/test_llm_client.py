import logging
from types import SimpleNamespace

import groq
import httpx
import pytest

from app.llm import client as client_module
from app.llm.client import FALLBACK_PERIOD, DailyLimitError, LLMClient, LLMError, MinuteLimitError, Usage

MINUTE = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01abc` on tokens per "
          "minute (TPM): Limit 8000, Used 7500, Requested 1500. Please try again in 7.5s.")
DAY = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01abc` on tokens per "
       "day (TPD): Limit 200000, Used 199500, Requested 1500. Please try again in 9m0s.")
HI = [{"role": "user", "content": "Hi"}]


def client_with(create, fallback_model: str | None = None) -> LLMClient:
    # The SDK client is replaced by an object whose chat.completions.create is `create`
    client = LLMClient(api_key="test", model="main-model", fallback_model=fallback_model)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client


def rate_limited(message: str):
    def create(**kwargs):
        response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
        raise groq.RateLimitError(message, response=response, body=None)
    return create


def answer(prompt_tokens: int, completion_tokens: int):
    def create(**kwargs):
        message = SimpleNamespace(content=f"Answer from {kwargs['model']}", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )
    return create


def by_model(behaviour: dict):
    # Answers or fails depending on the requested model, and remembers which models were asked
    def create(**kwargs):
        create.asked.append(kwargs["model"])
        return behaviour[kwargs["model"]](**kwargs)
    create.asked = []
    return create


def test_per_minute_limit():
    with pytest.raises(MinuteLimitError):
        client_with(rate_limited(MINUTE)).chat(HI)


def test_daily_limit():
    with pytest.raises(DailyLimitError) as error:
        client_with(rate_limited(DAY)).chat(HI)
    # The API maps every LLMError to 503, so the daily limit needs no extra handling there
    assert isinstance(error.value, LLMError)


def test_rate_limit_is_logged_without_the_organization_id(caplog):
    with caplog.at_level(logging.WARNING), pytest.raises(DailyLimitError):
        client_with(rate_limited(DAY)).chat(HI)
    assert "tokens per day" in caplog.text
    assert "org_01abc" not in caplog.text


def test_usage_adds_up_over_calls():
    client = client_with(answer(prompt_tokens=100, completion_tokens=20))
    client.chat(HI)
    client.chat([{"role": "user", "content": "Hi again"}])
    assert client.usage == Usage(calls=2, prompt_tokens=200, completion_tokens=40)


def test_fallback_model_answers_when_the_daily_limit_is_used_up():
    create = by_model({"main-model": rate_limited(DAY), "small-model": answer(10, 5)})
    assert client_with(create, fallback_model="small-model").chat(HI) == "Answer from small-model"
    assert create.asked == ["main-model", "small-model"]


def test_fallback_model_keeps_answering_without_asking_the_main_model_again():
    create = by_model({"main-model": rate_limited(DAY), "small-model": answer(10, 5)})
    client = client_with(create, fallback_model="small-model")
    client.chat(HI)
    client.chat(HI)
    assert create.asked == ["main-model", "small-model", "small-model"]


def test_main_model_is_asked_again_after_the_fallback_period(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock[0])
    create = by_model({"main-model": rate_limited(DAY), "small-model": answer(10, 5)})
    client = client_with(create, fallback_model="small-model")
    client.chat(HI)
    clock[0] += FALLBACK_PERIOD + 1
    client.chat(HI)
    assert create.asked == ["main-model", "small-model", "main-model", "small-model"]


def test_per_minute_limit_does_not_switch_models():
    create = by_model({"main-model": rate_limited(MINUTE), "small-model": answer(10, 5)})
    with pytest.raises(MinuteLimitError):
        client_with(create, fallback_model="small-model").chat(HI)
    assert create.asked == ["main-model"]


def test_fallback_model_with_its_own_daily_limit_used_up():
    create = by_model({"main-model": rate_limited(DAY), "small-model": rate_limited(DAY)})
    with pytest.raises(DailyLimitError):
        client_with(create, fallback_model="small-model").chat(HI)
