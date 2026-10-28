import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import groq
from groq.types.chat import ChatCompletionMessage

from app.core.config import get_settings

logger = logging.getLogger(__name__)

Message = dict[str, Any]


class LLMError(RuntimeError):
    """The LLM could not give an answer; the message is safe to show to the user."""


class ToolCallError(LLMError):
    """The model generated a tool call that the provider rejected (unknown tool or invalid arguments)."""


class MinuteLimitError(LLMError):
    """The provider's per-minute limit was hit: the same request works again within a minute."""


class DailyLimitError(LLMError):
    """The provider's daily limit is used up: retrying soon does not help."""


# The provider's rate limit message names the organization; the logs leave its id out
ORGANIZATION_ID = re.compile(r"org_[A-Za-z0-9]+")
# After the main model's daily limit is hit, the fallback model answers for this long (seconds)
# without asking the main model again: every such try costs a rejected request first
FALLBACK_PERIOD = 600.0


def _error_code(error: groq.APIStatusError) -> str | None:
    body = error.body if isinstance(error.body, dict) else {}
    return body.get("error", {}).get("code")


@dataclass
class Usage:
    """Calls and tokens since the client was created; the evaluation reads the difference per question."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        fallback_model: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        # The SDK itself retries connection errors, 408, 409, 429 and 5xx with exponential backoff
        self.client = groq.Groq(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model
        # Answers when the main model's daily limit is used up; it has its own limits at the provider
        self.fallback_model = fallback_model
        self._fallback_until = 0.0
        self.usage = Usage()

    def chat(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        message = self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        if not message.content:
            raise LLMError("The LLM returned an empty answer")
        return message.content

    def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatCompletionMessage:
        # tool_choice: "auto", "none", or {"type": "function", "function": {"name": ...}} to force one tool
        request = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            request |= {"tools": tools, "tool_choice": tool_choice}
        if self.fallback_model and time.monotonic() < self._fallback_until:
            return self._create(self.fallback_model, request)
        try:
            return self._create(self.model, request)
        except DailyLimitError:
            if not self.fallback_model:
                raise
            logger.warning(
                "Daily limit of %s used up: %s answers for the next %.0f minutes",
                self.model, self.fallback_model, FALLBACK_PERIOD / 60,
            )
            self._fallback_until = time.monotonic() + FALLBACK_PERIOD
            return self._create(self.fallback_model, request)

    def _create(self, model: str, request: dict[str, Any]) -> ChatCompletionMessage:
        try:
            response = self.client.chat.completions.create(model=model, **request)
        except groq.AuthenticationError as e:
            raise LLMError("The LLM provider rejected the API key") from e
        except groq.RateLimitError as e:
            # The message says which limit was hit, e.g. "tokens per minute (TPM)" or "tokens per day (TPD)"
            message = ORGANIZATION_ID.sub("org_***", str(e))
            logger.warning("Rate limit: %s", message)
            if "per day" in message:
                raise DailyLimitError("The LLM's daily limit is used up, please try again later") from e
            raise MinuteLimitError("LLM rate limit reached, please try again in a minute") from e
        except groq.APIConnectionError as e:
            raise LLMError("Could not reach the LLM provider") from e
        except groq.BadRequestError as e:
            if _error_code(e) == "tool_use_failed":
                raise ToolCallError("The LLM made an invalid tool call") from e
            raise LLMError("The LLM provider rejected the request (HTTP 400)") from e
        except groq.APIStatusError as e:
            raise LLMError(f"The LLM provider returned an error (HTTP {e.status_code})") from e

        choice = response.choices[0]
        usage = response.usage
        self.usage.calls += 1
        self.usage.prompt_tokens += usage.prompt_tokens
        self.usage.completion_tokens += usage.completion_tokens
        logger.info(
            "LLM call (%s): %d prompt + %d completion tokens", model, usage.prompt_tokens, usage.completion_tokens
        )
        if choice.finish_reason == "length":
            logger.warning("LLM answer was cut off at max_tokens=%d", request["max_tokens"])
        return choice.message


@lru_cache
def get_llm() -> LLMClient:
    settings = get_settings()
    return LLMClient(
        api_key=settings.groq_api_key.get_secret_value(),
        model=settings.groq_model,
        fallback_model=settings.groq_fallback_model,
    )
