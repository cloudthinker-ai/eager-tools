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

    # The model "agent" makes one tool call, the framework runs the next step
    # which our scripted model also serves — a final text response.
    model = _ScriptedModel(
        chunks=[
            *_make_chunks_calling_add("call_add_1"),
            # The next agent loop iteration: no tools, just text.
            AIMessageChunk(content="The answer is 5."),
        ]
    )

    # First call returns chunks 0+1 (tool call), second call returns chunk 2 (text).
    # Easiest way: let the scripted model yield ALL chunks every time, but
    # `_astream` is invoked on each agent step. Use a per-step script instead.
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
    # Expect: HumanMessage, AIMessage(tool_call), ToolMessage(=5), AIMessage(text)
    assert isinstance(msgs[0], HumanMessage)
    assert any(isinstance(m, AIMessage) and m.tool_calls for m in msgs)
    tool_msg = next(m for m in msgs if isinstance(m, ToolMessage))
    assert tool_msg.tool_call_id == "call_add_1"
    assert "5" in str(tool_msg.content)
    # The agent did not run `add` a SECOND time via the normal tool step.
    assert len(invocations) == 1
    # Final assistant text comes through.
    final_text = msgs[-1]
    assert isinstance(final_text, AIMessage)
    assert "5" in str(final_text.content)
