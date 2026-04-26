"""AnthropicEagerStream replay tests.

Tests feed synthetic events (SimpleNamespace, no SDK import) to keep CI fast,
deterministic, and offline. Source of truth: METHOD.md §3 + the core
SealDetector contract.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from eager_tools import SealEvent, Tool, ToolCall
from fixtures import (
    content_block_stop,
    input_json_delta,
    message_start,
    message_stop,
    ping,
    text_block_start,
    text_delta,
    tool_use_start,
)

from eager_tools_anthropic import AnthropicEagerStream
from eager_tools_anthropic.chunks import normalize_event


class FakeTool:
    def __init__(self, name: str, *, idempotent: bool = True, delay: float = 0.0) -> None:
        self.name = name
        self.idempotent = idempotent
        self._delay = delay

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        if self._delay:
            await asyncio.sleep(self._delay)
        return {"name": self.name, "args": arguments}


async def _async_iter(items: list[Any]) -> AsyncIterator[Any]:
    for it in items:
        yield it


def test_tool_use_block_start_emits_first_chunk() -> None:
    """`content_block_start{type=tool_use, id, name}` → chunk with id + name populated."""
    chunk = normalize_event(tool_use_start(0, "toolu_A", "read_file"))
    assert chunk is not None
    assert chunk.tool_call_id == "toolu_A"
    assert chunk.name == "read_file"
    assert chunk.index == 0
    assert chunk.args_delta == ""


def test_input_json_delta_emits_args_chunk() -> None:
    """`content_block_delta{type=input_json_delta}` → chunk with args_delta, no id."""
    chunk = normalize_event(input_json_delta(0, '{"path":'))
    assert chunk is not None
    assert chunk.tool_call_id is None
    assert chunk.index == 0
    assert chunk.args_delta == '{"path":'


def test_non_tool_events_return_none() -> None:
    """Text deltas, message metadata, ping, content_block_stop are ignored."""
    assert normalize_event(text_block_start(0)) is None
    assert normalize_event(text_delta(0, "hello")) is None
    assert normalize_event(content_block_stop(0)) is None
    assert normalize_event(message_start()) is None
    assert normalize_event(message_stop()) is None
    assert normalize_event(ping()) is None


async def test_end_to_end_two_tool_sequence() -> None:
    """Replay two sequential tool blocks; expect 2 tool_sealed + 1 message_complete."""
    raw = [
        tool_use_start(0, "A", "read_file"),
        input_json_delta(0, '{"path":"/a"}'),
        content_block_stop(0),
        tool_use_start(1, "B", "http_get"),
        input_json_delta(1, '{"url":"/b"}'),
        content_block_stop(1),
        message_stop(),
    ]
    tools: dict[str, Tool] = {
        "read_file": FakeTool("read_file"),
        "http_get": FakeTool("http_get"),
    }
    stream = AnthropicEagerStream(_async_iter(raw), tools=tools)
    seals = [ev async for ev in stream.events()]

    tool_seals = [s for s in seals if s.kind == "tool_sealed"]
    assert len(tool_seals) == 2
    assert tool_seals[0].tool_call is not None
    assert tool_seals[0].tool_call.tool_call_id == "A"
    assert tool_seals[0].tool_call.arguments == {"path": "/a"}
    assert tool_seals[1].tool_call is not None
    assert tool_seals[1].tool_call.tool_call_id == "B"
    assert tool_seals[1].tool_call.arguments == {"url": "/b"}
    assert seals[-1].kind == "message_complete"

    results = {call.tool_call_id: res async for call, res in stream.results()}
    assert results == {
        "A": {"name": "read_file", "args": {"path": "/a"}},
        "B": {"name": "http_get", "args": {"url": "/b"}},
    }


async def test_text_only_message_yields_message_complete_only() -> None:
    """A tool-less stream still terminates cleanly with one message_complete."""
    raw = [
        message_start(),
        text_block_start(0),
        text_delta(0, "hello "),
        text_delta(0, "world"),
        content_block_stop(0),
        message_stop(),
    ]
    stream = AnthropicEagerStream(_async_iter(raw), tools={})
    seals = [ev async for ev in stream.events()]
    assert len(seals) == 1
    assert seals[0].kind == "message_complete"
    results = [item async for item in stream.results()]
    assert results == []


async def test_mixed_text_and_tool_stream_seals_correctly() -> None:
    """Text deltas interspersed with tool blocks don't disrupt sealing."""
    raw = [
        message_start(),
        text_block_start(0),
        text_delta(0, "Calling tool now."),
        content_block_stop(0),
        tool_use_start(1, "T", "noop"),
        input_json_delta(1, "{}"),
        content_block_stop(1),
        message_stop(),
    ]
    stream = AnthropicEagerStream(_async_iter(raw), tools={"noop": FakeTool("noop")})
    seals = [ev async for ev in stream.events()]
    tool_seals = [s for s in seals if s.kind == "tool_sealed"]
    assert len(tool_seals) == 1
    assert tool_seals[0].tool_call is not None
    assert tool_seals[0].tool_call.name == "noop"
    assert tool_seals[0].tool_call.arguments == {}


async def test_cancel_releases_in_flight_tools() -> None:
    """Cancelling the events() task during a slow tool releases pool tasks cleanly."""
    started = asyncio.Event()

    class BlockingTool:
        name = "block"
        idempotent = True

        async def __call__(self, arguments: dict[str, Any]) -> Any:
            del arguments
            started.set()
            await asyncio.sleep(60)  # cancelled before this returns

    async def slow_source() -> AsyncIterator[Any]:
        yield tool_use_start(0, "X", "block")
        yield input_json_delta(0, "{}")
        yield content_block_stop(0)
        yield tool_use_start(1, "Y", "block")  # seals X → dispatches it
        # Pause forever so the message_stop branch never runs.
        await asyncio.sleep(60)

    tools: dict[str, Tool] = {"block": BlockingTool()}
    stream = AnthropicEagerStream(slow_source(), tools=tools)

    seals: list[Any] = []

    async def drain() -> None:
        async for ev in stream.events():
            seals.append(ev)

    task = asyncio.create_task(drain())
    await started.wait()
    pool = stream._pool  # pyright: ignore[reportPrivateUsage]
    assert pool.in_flight == 1

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # The except handler in events() ran cancel_all + close.
    assert pool.in_flight == 0
    assert any(s.kind == "tool_sealed" for s in seals)

    results = [item async for item in stream.results()]
    assert isinstance(results, list)


async def test_dispatch_unknown_tool_surfaces_in_results() -> None:
    """A sealed tool with no matching impl surfaces as KeyError in results()."""
    raw = [
        tool_use_start(0, "X", "missing"),
        input_json_delta(0, "{}"),
        content_block_stop(0),
        message_stop(),
    ]
    stream = AnthropicEagerStream(_async_iter(raw), tools={})
    _ = [ev async for ev in stream.events()]
    results: list[tuple[ToolCall, Any]] = [item async for item in stream.results()]
    assert len(results) == 1
    call, exc = results[0]
    assert call.name == "missing"
    assert isinstance(exc, KeyError)


async def test_observability_on_seal_fires_for_every_seal() -> None:
    """`on_seal` fires once per tool_sealed plus once for message_complete."""

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

    raw = [
        tool_use_start(0, "A", "read_file"),
        input_json_delta(0, '{"path":"/a"}'),
        content_block_stop(0),
        tool_use_start(1, "B", "http_get"),
        input_json_delta(1, '{"url":"/b"}'),
        content_block_stop(1),
        message_stop(),
    ]
    tools: dict[str, Tool] = {
        "read_file": FakeTool("read_file"),
        "http_get": FakeTool("http_get"),
    }
    observer = Counter()
    stream = AnthropicEagerStream(_async_iter(raw), tools=tools, observability=observer)
    _ = [ev async for ev in stream.events()]
    _ = [item async for item in stream.results()]

    # 2 tool_sealed + 1 message_complete
    assert len(observer.seals) == 3
    kinds = [s.kind for s in observer.seals]
    assert kinds == ["tool_sealed", "tool_sealed", "message_complete"]
    for seal in observer.seals[:2]:
        assert seal.seal_latency_ms is not None
        assert seal.seal_latency_ms >= 0.0
    assert observer.dispatch_starts == 2
    assert observer.dispatch_ends == [None, None]


async def test_observer_on_seal_exception_does_not_break_stream() -> None:
    """A broken observer must not abort the stream — adapter wraps `on_seal`
    in `contextlib.suppress(Exception)` to match `ExecutorPool._safe_hook`.
    """

    class BrokenObserver:
        def on_seal(self, event: SealEvent) -> None:
            del event
            raise RuntimeError("observer broken")

        def on_dispatch_start(self, call: ToolCall) -> None:
            del call

        def on_dispatch_end(self, call: ToolCall, error: Exception | None) -> None:
            del call, error

    raw = [
        tool_use_start(0, "A", "read_file"),
        input_json_delta(0, '{"path":"/a"}'),
        content_block_stop(0),
        message_stop(),
    ]
    tools: dict[str, Tool] = {"read_file": FakeTool("read_file")}
    stream = AnthropicEagerStream(_async_iter(raw), tools=tools, observability=BrokenObserver())

    seals = [ev async for ev in stream.events()]
    assert any(s.kind == "tool_sealed" for s in seals)
    assert seals[-1].kind == "message_complete"
    results = {call.tool_call_id: res async for call, res in stream.results()}
    assert results == {"A": {"name": "read_file", "args": {"path": "/a"}}}


async def test_malformed_json_does_not_crash_stream_and_surfaces_in_results() -> None:
    """Bad JSON args from the model produce an error in results() but don't
    abort the stream — a subsequent well-formed tool still runs."""
    import json

    raw = [
        message_start(),
        tool_use_start(0, "BAD", "read_file"),
        input_json_delta(0, '{"broken'),  # never closes
        content_block_stop(0),
        tool_use_start(1, "OK", "read_file"),
        input_json_delta(1, '{"path":"/etc"}'),
        content_block_stop(1),
        message_stop(),
    ]
    tools: dict[str, Tool] = {"read_file": FakeTool("read_file")}
    stream = AnthropicEagerStream(_async_iter(raw), tools=tools)

    seal_events = [ev async for ev in stream.events() if ev.kind == "tool_sealed"]
    assert len(seal_events) == 2
    bad_seal = next(s for s in seal_events if s.tool_call and s.tool_call.tool_call_id == "BAD")
    assert isinstance(bad_seal.parse_error, json.JSONDecodeError)

    results = {r[0].tool_call_id: r[1] async for r in stream.results()}
    assert isinstance(results["BAD"], json.JSONDecodeError)
    assert results["OK"] == {"name": "read_file", "args": {"path": "/etc"}}
