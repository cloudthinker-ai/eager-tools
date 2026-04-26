"""OpenAIEagerStream replay tests.

Tests feed synthetic chunks (SimpleNamespace, no SDK import) to keep CI fast,
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
    chunk_with_tool_calls,
    finish,
    text_chunk,
    tool_call_args,
    tool_call_first,
)

from eager_tools_openai import OpenAIEagerStream
from eager_tools_openai.chunks import normalize_chunk


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


def test_tool_calls_delta_first_emits_id_name() -> None:
    """First delta for a tool_call slot carries `id` + `function.name`."""
    chunk = chunk_with_tool_calls([tool_call_first(0, "call_A", "read_file")])
    out = normalize_chunk(chunk)
    assert len(out) == 1
    assert out[0].tool_call_id == "call_A"
    assert out[0].name == "read_file"
    assert out[0].index == 0
    assert out[0].args_delta == ""


def test_tool_calls_delta_args_routes_by_index() -> None:
    """Subsequent args delta lacks `id` — normalizer routes by `index`."""
    chunk = chunk_with_tool_calls([tool_call_args(0, '{"path":')])
    out = normalize_chunk(chunk)
    assert len(out) == 1
    assert out[0].tool_call_id is None
    assert out[0].index == 0
    assert out[0].args_delta == '{"path":'


def test_text_only_chunks_yield_no_normalized_chunks() -> None:
    """Text chunks (no tool_calls) and finish-only chunks return [] from normalize."""
    assert normalize_chunk(text_chunk("hello")) == []
    assert normalize_chunk(finish("stop")) == []
    assert normalize_chunk(finish("tool_calls")) == []


def test_multiple_tool_calls_in_one_chunk() -> None:
    """One chunk can carry deltas for several tool slots simultaneously."""
    chunk = chunk_with_tool_calls(
        [
            tool_call_first(0, "A", "fa"),
            tool_call_first(1, "B", "fb"),
        ]
    )
    out = normalize_chunk(chunk)
    assert len(out) == 2
    assert out[0].tool_call_id == "A"
    assert out[1].tool_call_id == "B"


async def test_end_to_end_two_tool_sequence() -> None:
    """Replay two sequential tool deltas; expect 2 tool_sealed + 1 message_complete."""
    raw = [
        chunk_with_tool_calls([tool_call_first(0, "call_A", "read_file")]),
        chunk_with_tool_calls([tool_call_args(0, '{"path":"/a"}')]),
        chunk_with_tool_calls([tool_call_first(1, "call_B", "http_get")]),
        chunk_with_tool_calls([tool_call_args(1, '{"url":"/b"}')]),
        finish("tool_calls"),
    ]
    tools: dict[str, Tool] = {
        "read_file": FakeTool("read_file"),
        "http_get": FakeTool("http_get"),
    }
    stream = OpenAIEagerStream(_async_iter(raw), tools=tools)
    seals = [ev async for ev in stream.events()]

    tool_seals = [s for s in seals if s.kind == "tool_sealed"]
    assert len(tool_seals) == 2
    assert tool_seals[0].tool_call is not None
    assert tool_seals[0].tool_call.tool_call_id == "call_A"
    assert tool_seals[0].tool_call.arguments == {"path": "/a"}
    assert tool_seals[1].tool_call is not None
    assert tool_seals[1].tool_call.tool_call_id == "call_B"
    assert tool_seals[1].tool_call.arguments == {"url": "/b"}
    assert seals[-1].kind == "message_complete"

    results = {call.tool_call_id: res async for call, res in stream.results()}
    assert results == {
        "call_A": {"name": "read_file", "args": {"path": "/a"}},
        "call_B": {"name": "http_get", "args": {"url": "/b"}},
    }


async def test_text_only_message_yields_message_complete_only() -> None:
    """A tool-less stream still terminates cleanly with one message_complete."""
    raw = [
        text_chunk("hello "),
        text_chunk("world"),
        finish("stop"),
    ]
    stream = OpenAIEagerStream(_async_iter(raw), tools={})
    seals = [ev async for ev in stream.events()]
    assert len(seals) == 1
    assert seals[0].kind == "message_complete"
    results = [item async for item in stream.results()]
    assert results == []


async def test_cancel_releases_in_flight_tools() -> None:
    """Cancelling the events() task during a slow tool releases pool tasks cleanly."""
    started = asyncio.Event()

    class BlockingTool:
        name = "block"
        idempotent = True

        async def __call__(self, arguments: dict[str, Any]) -> Any:
            del arguments
            started.set()
            await asyncio.sleep(60)

    async def slow_source() -> AsyncIterator[Any]:
        yield chunk_with_tool_calls([tool_call_first(0, "X", "block")])
        yield chunk_with_tool_calls([tool_call_args(0, "{}")])
        yield chunk_with_tool_calls([tool_call_first(1, "Y", "block")])  # seals X
        await asyncio.sleep(60)

    tools: dict[str, Tool] = {"block": BlockingTool()}
    stream = OpenAIEagerStream(slow_source(), tools=tools)

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

    assert pool.in_flight == 0
    assert any(s.kind == "tool_sealed" for s in seals)

    results = [item async for item in stream.results()]
    assert isinstance(results, list)


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
        chunk_with_tool_calls([tool_call_first(0, "call_A", "read_file")]),
        chunk_with_tool_calls([tool_call_args(0, '{"path":"/a"}')]),
        chunk_with_tool_calls([tool_call_first(1, "call_B", "http_get")]),
        chunk_with_tool_calls([tool_call_args(1, '{"url":"/b"}')]),
        finish("tool_calls"),
    ]
    tools: dict[str, Tool] = {
        "read_file": FakeTool("read_file"),
        "http_get": FakeTool("http_get"),
    }
    observer = Counter()
    stream = OpenAIEagerStream(_async_iter(raw), tools=tools, observability=observer)
    _ = [ev async for ev in stream.events()]
    _ = [item async for item in stream.results()]

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
        chunk_with_tool_calls([tool_call_first(0, "call_A", "read_file")]),
        chunk_with_tool_calls([tool_call_args(0, '{"path":"/a"}')]),
        finish("tool_calls"),
    ]
    tools: dict[str, Tool] = {"read_file": FakeTool("read_file")}
    stream = OpenAIEagerStream(_async_iter(raw), tools=tools, observability=BrokenObserver())

    seals = [ev async for ev in stream.events()]
    assert any(s.kind == "tool_sealed" for s in seals)
    assert seals[-1].kind == "message_complete"
    results = {call.tool_call_id: res async for call, res in stream.results()}
    assert results == {"call_A": {"name": "read_file", "args": {"path": "/a"}}}


async def test_dispatch_unknown_tool_surfaces_in_results() -> None:
    """A sealed tool with no matching impl surfaces as KeyError in results()."""
    raw = [
        chunk_with_tool_calls([tool_call_first(0, "X", "missing")]),
        chunk_with_tool_calls([tool_call_args(0, "{}")]),
        finish("tool_calls"),
    ]
    stream = OpenAIEagerStream(_async_iter(raw), tools={})
    _ = [ev async for ev in stream.events()]
    results: list[tuple[ToolCall, Any]] = [item async for item in stream.results()]
    assert len(results) == 1
    call, exc = results[0]
    assert call.name == "missing"
    assert isinstance(exc, KeyError)
