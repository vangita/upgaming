import logging
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


def _error_code(error: groq.APIStatusError) -> str | None:
    body = error.body if isinstance(error.body, dict) else {}
    return body.get("error", {}).get("code")


class LLMClient:
    def __init__(self, api_key: str, model: str, timeout: float = 30.0, max_retries: int = 3) -> None:
        # The SDK itself retries connection errors, 408, 409, 429 and 5xx with exponential backoff
        self.client = groq.Groq(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model

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
        tool_args = {"tools": tools, "tool_choice": tool_choice} if tools else {}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **tool_args,
            )
        except groq.AuthenticationError as e:
            raise LLMError("The LLM provider rejected the API key") from e
        except groq.RateLimitError as e:
            raise LLMError("LLM rate limit reached, please try again in a minute") from e
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
        logger.info(
            "LLM call: %d prompt + %d completion tokens", usage.prompt_tokens, usage.completion_tokens
        )
        if choice.finish_reason == "length":
            logger.warning("LLM answer was cut off at max_tokens=%d", max_tokens)
        return choice.message


@lru_cache
def get_llm() -> LLMClient:
    settings = get_settings()
    return LLMClient(api_key=settings.groq_api_key.get_secret_value(), model=settings.groq_model)
