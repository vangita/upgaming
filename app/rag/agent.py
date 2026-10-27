import json
import logging

from groq.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from pydantic import ValidationError

from app.llm.client import LLMClient, Message, ToolCallError, get_llm
from app.rag.pipeline import Conflict, RAGAnswer
from app.rag.prompts import FINAL_INSTRUCTION, NO_ANSWER, agent_system_prompt, build_messages
from app.rag.tools import (
    SEARCH_TOOL,
    SUBMIT_ANSWER,
    SUBMIT_ANSWER_TOOL,
    SourceList,
    format_search_result,
)
from app.retrieval.retriever import Retriever, SearchMode

logger = logging.getLogger(__name__)

FORCE_SUBMIT = {"type": "function", "function": {"name": SUBMIT_ANSWER}}


def _arguments(call: ChatCompletionMessageToolCall) -> dict:
    try:
        arguments = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        logger.warning("Tool call %s has invalid JSON arguments", call.function.name)
        return {}
    return arguments if isinstance(arguments, dict) else {}


def _assistant_turn(message: ChatCompletionMessage) -> Message:
    # Rebuilt by hand: the SDK object also carries fields (e.g. `reasoning`)
    # that the API does not accept back as input
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ],
    }


class InvestigationAgent:
    """Multi-step RAG: the LLM reads the sources, searches for what is missing, then submits a structured answer."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient | None = None,
        top_k: int = 6,
        search_k: int = 3,
        max_searches: int = 2,
    ) -> None:
        self.retriever = retriever
        self.llm = llm or get_llm()
        self.top_k = top_k
        self.search_k = search_k
        self.max_searches = max_searches

    def answer(self, question: str, mode: SearchMode = "hybrid") -> RAGAnswer:
        sources = SourceList()
        initial = self.retriever.search(question, mode=mode, limit=self.top_k)
        if not initial:
            logger.info("No sources found, answering without calling the LLM")
            return RAGAnswer.build(question, NO_ANSWER, [])
        sources.add(initial)

        messages = build_messages(question, initial, system=agent_system_prompt(self.max_searches))
        searches: list[str] = []

        while len(searches) < self.max_searches:
            try:
                message = self.llm.complete(messages, tools=[SEARCH_TOOL, SUBMIT_ANSWER_TOOL])
            except ToolCallError:
                logger.warning("The LLM made an invalid tool call, asking for the final answer")
                break

            if not message.tool_calls:
                # The model answered in plain text instead of calling submit_answer
                return RAGAnswer.build(question, message.content or "", sources.hits, searches=searches)
            submit = next((c for c in message.tool_calls if c.function.name == SUBMIT_ANSWER), None)
            if submit:
                return self._finish(question, submit, sources, searches)

            messages.append(_assistant_turn(message))
            for call in message.tool_calls:
                query = str(_arguments(call).get("query", "")).strip()
                searches.append(query)
                found = sources.add(self.retriever.search(query, mode=mode, limit=self.search_k)) if query else []
                logger.info("Agent search %d: %r -> %d sources", len(searches), query, len(found))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": format_search_result(found)})

        # Out of searches (or the model broke a tool call): only submit_answer is offered now
        messages.append({"role": "user", "content": FINAL_INSTRUCTION})
        message = self.llm.complete(messages, tools=[SUBMIT_ANSWER_TOOL], tool_choice=FORCE_SUBMIT)
        if not message.tool_calls:
            return RAGAnswer.build(question, message.content or "", sources.hits, searches=searches)
        return self._finish(question, message.tool_calls[0], sources, searches)

    def _finish(
        self, question: str, call: ChatCompletionMessageToolCall, sources: SourceList, searches: list[str]
    ) -> RAGAnswer:
        arguments = _arguments(call)
        conflicts = []
        for item in arguments.get("conflicts") or []:
            try:
                conflicts.append(Conflict.model_validate(item))
            except ValidationError:
                logger.warning("Ignoring a malformed conflict: %r", item)
        listed = [n for n in arguments.get("sources") or [] if isinstance(n, int)]
        return RAGAnswer.build(
            question,
            str(arguments.get("answer", "")),
            sources.hits,
            answered=bool(arguments.get("answered", True)),
            conflicts=conflicts,
            searches=searches,
            listed_sources=listed,
        )
