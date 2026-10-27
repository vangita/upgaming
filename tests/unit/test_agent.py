import json

from groq.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from groq.types.chat.chat_completion_message_tool_call import Function

from app.ingestion.models import Chunk
from app.llm.client import ToolCallError
from app.rag.agent import FORCE_SUBMIT, InvestigationAgent
from app.rag.prompts import FINAL_INSTRUCTION, NO_ANSWER
from app.storage.qdrant_store import SearchHit

QUESTION = "question?"


def hit(title: str) -> SearchHit:
    text = f"Text about {title}."
    chunk = Chunk(text=text, title=title, source="test.md", file_format="md",
                  chunk_index=0, char_start=0, char_end=len(text))
    return SearchHit(id=title, score=0.5, chunk=chunk)


class FakeRetriever:
    """Returns the hits listed for each query; any other query finds nothing."""

    def __init__(self, results: dict[str, list[SearchHit]]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query, mode="hybrid", limit=10, source=None):
        self.queries.append(query)
        return self.results.get(query, [])[:limit]


class FakeLLM:
    """Returns the scripted replies in order; an exception in the script is raised instead."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, messages, tools=None, tool_choice="auto", **kwargs):
        self.calls.append({
            "messages": list(messages),
            "tools": [tool["function"]["name"] for tool in tools or []],
            "tool_choice": tool_choice,
        })
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def tool_call(name: str, arguments: dict | str, call_id: str = "call_1") -> ChatCompletionMessageToolCall:
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return ChatCompletionMessageToolCall(id=call_id, type="function", function=Function(name=name, arguments=arguments))


def reply(*calls: ChatCompletionMessageToolCall, content: str | None = None) -> ChatCompletionMessage:
    return ChatCompletionMessage(role="assistant", content=content, tool_calls=list(calls) or None)


def search(query: str, call_id: str = "call_1") -> ChatCompletionMessage:
    return reply(tool_call("search", {"query": query}, call_id))


def submit(answer: str, sources=(), answered=True, conflicts=()) -> ChatCompletionMessage:
    arguments = {"answer": answer, "sources": list(sources), "answered": answered, "conflicts": list(conflicts)}
    return reply(tool_call("submit_answer", arguments))


def make_agent(llm, results=None, **kwargs) -> InvestigationAgent:
    if results is None:
        results = {QUESTION: [hit("A"), hit("B")]}
    return InvestigationAgent(FakeRetriever(results), llm=llm, **kwargs)


def cited(result) -> list[tuple[int, str]]:
    return [(c.number, c.hit.chunk.title) for c in result.citations]


def test_answers_from_the_first_sources_without_searching():
    llm = FakeLLM(submit("Fact [2].", sources=[2]))
    result = make_agent(llm).answer(QUESTION)
    assert result.answered and result.searches == []
    assert cited(result) == [(2, "B")]
    assert llm.calls[0]["tools"] == ["search", "submit_answer"]


def test_search_results_continue_the_numbering():
    llm = FakeLLM(search("more about C"), submit("A [1]. C [3].", sources=[1, 3]))
    agent = make_agent(llm, {QUESTION: [hit("A"), hit("B")], "more about C": [hit("B"), hit("C")]})
    result = agent.answer(QUESTION)
    assert agent.retriever.queries == [QUESTION, "more about C"]
    assert result.searches == ["more about C"]
    assert [h.chunk.title for h in result.sources] == ["A", "B", "C"]
    assert cited(result) == [(1, "A"), (3, "C")]


def test_llm_gets_the_search_result_as_a_tool_message():
    llm = FakeLLM(search("more about C", call_id="call_7"), submit("C [3].", sources=[3]))
    make_agent(llm, {QUESTION: [hit("A"), hit("B")], "more about C": [hit("B"), hit("C")]}).answer(QUESTION)
    assistant, tool = llm.calls[1]["messages"][-2:]
    assert assistant["role"] == "assistant" and assistant["tool_calls"][0]["id"] == "call_7"
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call_7"
    assert "[2] B (already shown above)" in tool["content"]
    assert "[3] C\nText about C." in tool["content"]


def test_only_submit_answer_is_offered_after_the_last_search():
    llm = FakeLLM(search("x"), search("y"), submit("A [1].", sources=[1]))
    result = make_agent(llm, max_searches=2).answer(QUESTION)
    assert result.searches == ["x", "y"]
    final = llm.calls[-1]
    assert final["tools"] == ["submit_answer"] and final["tool_choice"] == FORCE_SUBMIT
    assert final["messages"][-1] == {"role": "user", "content": FINAL_INSTRUCTION}


def test_invalid_tool_call_skips_to_the_final_answer():
    llm = FakeLLM(ToolCallError("invalid tool call"), submit("A [1].", sources=[1]))
    result = make_agent(llm).answer(QUESTION)
    assert result.answered and cited(result) == [(1, "A")]
    assert len(llm.calls) == 2 and llm.calls[1]["tools"] == ["submit_answer"]


def test_plain_text_reply_is_used_as_the_answer():
    llm = FakeLLM(reply(content="Fact【1】."))
    result = make_agent(llm).answer(QUESTION)
    assert result.answer == "Fact[1]." and cited(result) == [(1, "A")]


def test_listed_sources_are_used_when_the_text_has_no_markers():
    llm = FakeLLM(submit("A fact without markers.", sources=[2, 2, 1]))
    result = make_agent(llm).answer(QUESTION)
    assert cited(result) == [(2, "B"), (1, "A")]


def test_abstention_has_no_citations():
    llm = FakeLLM(submit("The sources do not say when C happened.", sources=[1, 2], answered=False))
    result = make_agent(llm).answer(QUESTION)
    assert not result.answered and result.citations == []


def test_broken_arguments_become_an_abstention():
    llm = FakeLLM(reply(tool_call("submit_answer", "{not json")))
    result = make_agent(llm).answer(QUESTION)
    assert result.answer == NO_ANSWER and not result.answered


def test_conflicts_keep_only_valid_entries_and_sources():
    conflicts = [
        {"description": "Dates differ", "sources": [1, 2, 9]},
        {"description": "No sources given"},
        {"description": "Only unknown sources", "sources": [9]},
    ]
    llm = FakeLLM(submit("A says 1990 [1], B says 1991 [2].", sources=[1, 2], conflicts=conflicts))
    result = make_agent(llm).answer(QUESTION)
    assert [(c.description, c.sources) for c in result.conflicts] == [("Dates differ", [1, 2])]


def test_no_sources_means_no_llm_call():
    llm = FakeLLM()
    result = make_agent(llm, results={}).answer(QUESTION)
    assert result.answer == NO_ANSWER and llm.calls == []
