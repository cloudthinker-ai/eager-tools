"""End-to-end smoke test: `create_agent(... middleware=[eager_middleware(...)])`.

We can't easily plug a real provider into CI, so we subclass `BaseChatModel`
with a scripted `astream` that emits a known `AIMessageChunk` sequence.

What this proves:
- `create_agent` accepts `EagerMiddleware` and routes through `awrap_model_call`.
- The middleware's returned `ModelResponse(result=[AIMessage, ToolMessage…])`
  is correctly committed to graph state via the `add_messages` reducer.
- The agent's normal tool step is bypassed for ids the middleware already resolved
  (they appear once in state, not twice).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.tools import tool

from eager_tools_langgraph import eager_middleware


class _ScriptedModel(BaseChatModel):
    """Minimal `BaseChatModel` stub: yields a fixed `AIMessageChunk` sequence."""

    chunks: list[AIMessageChunk]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[Any], stop: Any = None, **_kw: Any) -> ChatResult:
        # Should not be called when astream is used, but required by ABC.
        merged: AIMessageChunk | None = None
        for c in self.chunks:
            merged = c if merged is None else merged + c
        return ChatResult(generations=[])  # unused — astream is the path

    async def _astream(  # type: ignore[override]
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        for chunk in self.chunks:
            yield ChatGenerationChunk(message=chunk)

    def bind_tools(self, tools: Any, **_kw: Any) -> _ScriptedModel:  # type: ignore[override]
        del tools
        return self


def _make_chunks_calling_add(call_id: str) -> list[AIMessageChunk]:
    """Two streamed chunks calling `add(a=2, b=3)`."""
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "add", "args": "", "id": call_id, "index": 0}],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": None, "args": '{"a":2,"b":3}', "id": None, "index": 0}],
        ),
    ]


@pytest.mark.asyncio
async def test_create_agent_with_eager_middleware_runs_tool_eagerly() -> None:
    """The middleware resolves `add(2,3)`, agent commits AI+ToolMessage in one step,
    then the model returns a final text-only response."""
    invocations: list[dict[str, Any]] = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # Eager wrapper: needs a Tool-protocol object (name + idempotent + __call__).
    class _AddEager:
        name = "add"
        idempotent = True

        async def __call__(self, args: dict[str, Any]) -> Any:
            invocations.append(args)
            return add.invoke(args)

    # First model step: tool call. Second step (after eager ToolMessage commits):
    # final text. The scripted model returns a different sequence per call.
    step_idx = {"i": 0}
    scripts = [
        _make_chunks_calling_add("call_add_1"),
        [AIMessageChunk(content="The answer is 5.")],
    ]

    class _SteppedModel(_ScriptedModel):
        async def _astream(self, *_a: Any, **_kw: Any) -> AsyncIterator[ChatGenerationChunk]:
            i = step_idx["i"]
            step_idx["i"] += 1
            for chunk in scripts[min(i, len(scripts) - 1)]:
                yield ChatGenerationChunk(message=chunk)

    model = _SteppedModel(chunks=[])

    agent = create_agent(
        model=model,
        tools=[add],
        middleware=[eager_middleware({"add": _AddEager()})],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("what is 2+3?")]})

    assert invocations == [{"a": 2, "b": 3}]

    msgs = result["messages"]
    # Expect: HumanMessage, AIMessage(tool_call=add(2,3)), ToolMessage(=5), AIMessage(text)
    assert isinstance(msgs[0], HumanMessage)
    ai_with_calls = next(m for m in msgs if isinstance(m, AIMessage) and m.tool_calls)
    assert ai_with_calls.tool_calls[0]["name"] == "add"
    assert ai_with_calls.tool_calls[0]["args"] == {"a": 2, "b": 3}
    tool_msg = next(m for m in msgs if isinstance(m, ToolMessage))
    assert tool_msg.tool_call_id == "call_add_1"
    assert "5" in str(tool_msg.content)
    # The agent did not run `add` a SECOND time via the normal tool step.
    assert len(invocations) == 1
    # Final assistant text comes through.
    final_text = msgs[-1]
    assert isinstance(final_text, AIMessage)
    assert "5" in str(final_text.content)


@pytest.mark.asyncio
async def test_create_agent_with_eager_middleware_recovers_from_malformed_tool_args() -> None:
    """A malformed tool call doesn't crash the agent — it surfaces as an
    error ToolMessage that the model can react to on the next turn."""
    invocations: list[dict[str, Any]] = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    class _AddEager:
        name = "add"
        idempotent = True

        async def __call__(self, args: dict[str, Any]) -> Any:
            invocations.append(args)
            return add.invoke(args)

    bad_chunks = [
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "add", "args": "", "id": "call_bad_1", "index": 0}],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": None, "args": '{"a":1, "b":', "id": None, "index": 0}],
        ),
    ]
    step_idx = {"i": 0}
    scripts = [
        bad_chunks,
        [AIMessageChunk(content="Sorry, I'll try again.")],
    ]

    class _SteppedModel(_ScriptedModel):
        async def _astream(self, *_a: Any, **_kw: Any) -> AsyncIterator[ChatGenerationChunk]:
            i = step_idx["i"]
            step_idx["i"] += 1
            for chunk in scripts[min(i, len(scripts) - 1)]:
                yield ChatGenerationChunk(message=chunk)

    model = _SteppedModel(chunks=[])

    agent = create_agent(
        model=model,
        tools=[add],
        middleware=[eager_middleware({"add": _AddEager()})],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("compute 1+?")]})

    # The eager wrapper was never called — the JSON didn't parse.
    assert invocations == []

    msgs = result["messages"]
    # Exactly one ToolMessage for `call_bad_1`, marked as an error.
    error_msgs = [m for m in msgs if isinstance(m, ToolMessage) and m.tool_call_id == "call_bad_1"]
    assert len(error_msgs) == 1
    assert error_msgs[0].status == "error"
    assert "JSONDecodeError" in str(error_msgs[0].content)

    # The agent moved on to the next model turn — final text is present.
    final_text = msgs[-1]
    assert isinstance(final_text, AIMessage)
    assert "try again" in str(final_text.content)
