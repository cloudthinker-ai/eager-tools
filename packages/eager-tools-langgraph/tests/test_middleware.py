"""Replay tests for `EagerMiddleware`.

We stub the `ModelRequest.model` with a `ScriptedStream` that yields a known
sequence of `AIMessageChunk`s, then drive the middleware directly via
`awrap_model_call`. No live model, no LangGraph runtime — all the eager-dispatch
behavior we care about is observable through `ModelResponse.result`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from eager_tools import SealEvent, ToolCall
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eager_tools_langgraph import EagerMiddleware

from .fixtures import ScriptedStream, parallel_tool_chunk, script, text_chunk, tool_chunk


class _RecordingTool:
    """Records every call. Returns a fixed payload after an optional sleep."""

    def __init__(
        self, name: str, *, idempotent: bool = True, payload: Any = None, sleep: float = 0.0
    ) -> None:
        self.name = name
        self.idempotent = idempotent
        self._payload = payload if payload is not None else {"ok": name}
        self._sleep = sleep
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        self.calls.append(arguments)
        if self._sleep:
            await asyncio.sleep(self._sleep)
        return self._payload


def _make_request(model: ScriptedStream) -> Any:
    """Minimal ModelRequest stand-in (only .model / .messages / .tools / .model_settings)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        model=model,
        messages=[HumanMessage("go")],
        tools=[],
        model_settings={},
    )


async def _run(mw: EagerMiddleware, model: ScriptedStream) -> Any:
    return await mw.awrap_model_call(_make_request(model), handler=None)


@pytest.mark.asyncio
async def test_single_idempotent_tool_dispatched_eagerly() -> None:
    """One tool, three streamed chunks, dispatched at finalize time."""
    tool = _RecordingTool("read_file", payload={"text": "hello"})
    mw = EagerMiddleware({"read_file": tool})
    model = script(
        tool_chunk(index=0, tool_id="call_1", name="read_file", args=""),
        tool_chunk(index=0, args='{"path":'),
        tool_chunk(index=0, args='"a.txt"}'),
    )

    response = await _run(mw, model)

    assert tool.calls == [{"path": "a.txt"}]
    assert len(response.result) == 2
    ai_msg, tool_msg = response.result
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.tool_calls[0]["name"] == "read_file"
    assert ai_msg.tool_calls[0]["args"] == {"path": "a.txt"}
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "call_1"
    assert tool_msg.name == "read_file"
    assert json.loads(str(tool_msg.content)) == {"text": "hello"}


@pytest.mark.asyncio
async def test_three_parallel_tools_overlap() -> None:
    """Three idempotent tools sealed sequentially — all dispatched, all drained."""
    tools = {f"t{i}": _RecordingTool(f"t{i}", payload={"i": i}, sleep=0.05) for i in range(3)}
    mw = EagerMiddleware(tools)

    chunks = []
    for i in range(3):
        chunks.append(tool_chunk(index=i, tool_id=f"call_{i}", name=f"t{i}", args=""))
        chunks.append(tool_chunk(index=i, args=f'{{"v":{i}}}'))
    model = script(*chunks)

    response = await _run(mw, model)

    assert all(t.calls == [{"v": int(t.name[1:])}] for t in tools.values())
    ai_msg = response.result[0]
    tool_msgs = response.result[1:]
    assert isinstance(ai_msg, AIMessage)
    assert {tc["id"] for tc in ai_msg.tool_calls} == {"call_0", "call_1", "call_2"}
    assert {m.tool_call_id for m in tool_msgs} == {"call_0", "call_1", "call_2"}
    # Order in result mirrors AIMessage.tool_calls order, not completion order.
    expected_order = [tc["id"] for tc in ai_msg.tool_calls]
    assert [m.tool_call_id for m in tool_msgs] == expected_order


@pytest.mark.asyncio
async def test_non_idempotent_tool_falls_through() -> None:
    """Mixed batch: idempotent dispatched eagerly, non-idempotent skipped."""
    safe = _RecordingTool("safe", idempotent=True, payload="ok")
    unsafe = _RecordingTool("send_email", idempotent=False, payload="sent")
    mw = EagerMiddleware({"safe": safe, "send_email": unsafe})
    model = script(
        tool_chunk(index=0, tool_id="call_safe", name="safe", args="{}"),
        tool_chunk(index=1, tool_id="call_unsafe", name="send_email", args='{"to":"x"}'),
    )

    response = await _run(mw, model)

    assert safe.calls == [{}]
    assert unsafe.calls == []  # never invoked by middleware

    ai_msg = response.result[0]
    tool_msgs = response.result[1:]
    assert isinstance(ai_msg, AIMessage)
    # AI message preserves BOTH tool calls — the agent's tool step picks up
    # the unsafe one.
    assert {tc["id"] for tc in ai_msg.tool_calls} == {"call_safe", "call_unsafe"}
    # But middleware emits only the eager (safe) ToolMessage.
    assert [m.tool_call_id for m in tool_msgs] == ["call_safe"]


@pytest.mark.asyncio
async def test_unknown_tool_yields_error_tool_message() -> None:
    """ExecutorPool's unknown-tool KeyError surfaces as a ToolMessage(status=error)."""
    mw = EagerMiddleware({})  # registry is empty
    model = script(
        tool_chunk(index=0, tool_id="call_x", name="ghost", args="{}"),
    )

    response = await _run(mw, model)

    ai_msg = response.result[0]
    tool_msgs = response.result[1:]
    assert isinstance(ai_msg, AIMessage)
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "error"
    assert tool_msgs[0].tool_call_id == "call_x"
    assert "KeyError" in str(tool_msgs[0].content) or "ghost" in str(tool_msgs[0].content)


@pytest.mark.asyncio
async def test_parallel_chunk_in_single_message() -> None:
    """A single AIMessageChunk carrying two tool_call_chunks at once."""
    a = _RecordingTool("a", payload="A")
    b = _RecordingTool("b", payload="B")
    mw = EagerMiddleware({"a": a, "b": b})
    model = script(
        parallel_tool_chunk(
            {"index": 0, "id": "call_a", "name": "a", "args": "{}"},
            {"index": 1, "id": "call_b", "name": "b", "args": "{}"},
        ),
    )

    response = await _run(mw, model)

    assert a.calls == [{}]
    assert b.calls == [{}]
    ai_msg = response.result[0]
    assert isinstance(ai_msg, AIMessage)
    assert {tc["id"] for tc in ai_msg.tool_calls} == {"call_a", "call_b"}


@pytest.mark.asyncio
async def test_observability_on_seal_fires_for_each_sealed_tool() -> None:
    """Observer's `on_seal` fires once per sealed tool. Middleware emits no
    `message_complete` SealEvent (returns ModelResponse instead)."""

    class Counter:
        def __init__(self) -> None:
            self.seals: list[SealEvent] = []
            self.dispatch_starts: int = 0
            self.dispatch_ends: list[Exception | None] = []

        def on_seal(self, event: SealEvent) -> None:
            self.seals.append(event)

        def on_dispatch_start(self, call: ToolCall) -> None:
            del call
            self.dispatch_starts += 1

        def on_dispatch_end(self, call: ToolCall, error: Exception | None) -> None:
            del call
            self.dispatch_ends.append(error)

    observer = Counter()
    a = _RecordingTool("a", payload="A")
    b = _RecordingTool("b", payload="B")
    mw = EagerMiddleware({"a": a, "b": b}, observability=observer)
    model = script(
        tool_chunk(index=0, tool_id="call_a", name="a", args="{}"),
        tool_chunk(index=1, tool_id="call_b", name="b", args="{}"),
    )

    await _run(mw, model)

    assert len(observer.seals) == 2
    assert all(s.kind == "tool_sealed" for s in observer.seals)
    assert {s.tool_call.tool_call_id for s in observer.seals if s.tool_call} == {
        "call_a",
        "call_b",
    }
    for seal in observer.seals:
        assert seal.seal_latency_ms is not None
        assert seal.seal_latency_ms >= 0.0
    assert observer.dispatch_starts == 2
    assert observer.dispatch_ends == [None, None]


@pytest.mark.asyncio
async def test_observer_on_seal_exception_does_not_break_middleware() -> None:
    """A broken observer must not abort the model step — middleware wraps
    `on_seal` in `contextlib.suppress(Exception)` inside `_handle_seal`.
    """

    class BrokenObserver:
        def on_seal(self, event: SealEvent) -> None:
            del event
            raise RuntimeError("observer broken")

        def on_dispatch_start(self, call: ToolCall) -> None:
            del call

        def on_dispatch_end(self, call: ToolCall, error: Exception | None) -> None:
            del call, error

    tool = _RecordingTool("read_file", payload={"text": "hello"})
    mw = EagerMiddleware({"read_file": tool}, observability=BrokenObserver())
    model = script(
        tool_chunk(index=0, tool_id="call_1", name="read_file", args='{"path":"a.txt"}'),
    )

    response = await _run(mw, model)

    assert tool.calls == [{"path": "a.txt"}]
    assert len(response.result) == 2
    ai_msg, tool_msg = response.result
    assert isinstance(ai_msg, AIMessage)
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_text_only_stream_emits_no_tool_messages() -> None:
    """Pure text response — middleware returns the AIMessage unchanged, no tool step."""
    mw = EagerMiddleware({})
    model = script(text_chunk("hello "), text_chunk("world"))

    response = await _run(mw, model)

    assert len(response.result) == 1
    ai_msg = response.result[0]
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == "hello world"
    assert ai_msg.tool_calls == []


@pytest.mark.asyncio
async def test_cancellation_propagates_and_cancels_in_flight() -> None:
    """If the awrap_model_call task is cancelled mid-stream, in-flight tools die too."""
    started = asyncio.Event()
    finished = asyncio.Event()

    class _SlowTool:
        name = "slow"
        idempotent = True

        async def __call__(self, args: dict[str, Any]) -> Any:
            started.set()
            try:
                await asyncio.sleep(60)
                finished.set()  # should never reach here
                return "done"
            except asyncio.CancelledError:
                raise

    async def _slow_stream():
        # Seal the tool early (a NEW tool_call_id arriving causes the previous
        # one to seal), then block forever so the middleware is still inside
        # the astream loop when we cancel it from the outside.
        yield tool_chunk(index=0, tool_id="call_slow", name="slow", args="{}")
        yield tool_chunk(index=1, tool_id="call_filler", name="filler", args="{}")
        await asyncio.sleep(60)

    class _Model:
        def astream(self, messages: Any, **_kw: Any) -> Any:
            del messages
            return _slow_stream()

    mw = EagerMiddleware({"slow": _SlowTool()})
    request = _make_request(_Model())  # type: ignore[arg-type]

    task = asyncio.create_task(mw.awrap_model_call(request, handler=None))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not finished.is_set()
