"""Synthetic bench harness — paces a fake Anthropic stream like a real provider.

The same `FakeStream` is consumed by all three dispatch modes (sequential,
parallel, eager) so the comparison is fair: identical chunk timing, identical
tool latencies, identical workload shape. Deterministic and free to run in CI.

Event shape mirrors `packages/eager-tools-anthropic/tests/fixtures.py`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """One tool the model will emit during the stream."""

    name: str
    delay_ms: float
    start_offset_ms: float
    args_json: str = '{"q": "x"}'
    arg_chunk_count: int = 4


@dataclass(frozen=True)
class Workload:
    """A reproducible bench workload."""

    name: str
    stream_duration_ms: float
    tools: list[ToolSpec] = field(default_factory=list[ToolSpec])

    @property
    def total_tool_seconds(self) -> float:
        return sum(t.delay_ms for t in self.tools) / 1000.0

    @property
    def slowest_tool_seconds(self) -> float:
        return max((t.delay_ms for t in self.tools), default=0.0) / 1000.0


class LatencyTool:
    """Idempotent fake tool that sleeps `delay_ms` then returns a marker."""

    def __init__(self, name: str, delay_ms: float) -> None:
        self.name = name
        self.idempotent = True
        self._delay_s = delay_ms / 1000.0

    async def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(self._delay_s)
        return {"name": self.name, "args": arguments, "ok": True}


def build_tools(workload: Workload) -> dict[str, LatencyTool]:
    """Map of name → LatencyTool keyed by tool name (deduped, last delay wins)."""
    return {t.name: LatencyTool(t.name, t.delay_ms) for t in workload.tools}


def _split_json(s: str, parts: int) -> list[str]:
    """Split `s` into `parts` non-empty pieces (last absorbs the remainder)."""
    parts = max(1, min(parts, len(s)))
    step = max(1, len(s) // parts)
    chunks = [s[i * step : (i + 1) * step] for i in range(parts - 1)]
    chunks.append(s[(parts - 1) * step :])
    return chunks


async def fake_stream(workload: Workload) -> AsyncIterator[Any]:
    """Yield Anthropic-shaped events paced like a real stream.

    Timing: each tool's `content_block_start` fires at `start_offset_ms` from
    stream open; its argument chunks are spread evenly between that offset and
    the next tool's start (or stream end). `message_stop` fires at
    `stream_duration_ms`.
    """
    yield SimpleNamespace(type="message_start")
    t0 = asyncio.get_event_loop().time()

    sorted_tools = sorted(enumerate(workload.tools), key=lambda x: x[1].start_offset_ms)
    boundaries: list[float] = [t.start_offset_ms for _, t in sorted_tools]
    boundaries.append(workload.stream_duration_ms)

    for slot, (orig_idx, tool) in enumerate(sorted_tools):
        block_idx = orig_idx
        await _sleep_until(t0, tool.start_offset_ms)
        yield SimpleNamespace(
            type="content_block_start",
            index=block_idx,
            content_block=SimpleNamespace(
                type="tool_use",
                id=f"toolu_{block_idx:02d}",
                name=tool.name,
            ),
        )

        next_boundary = boundaries[slot + 1]
        chunk_strs = _split_json(tool.args_json, tool.arg_chunk_count)
        span = max(0.0, next_boundary - tool.start_offset_ms)
        per_chunk = span / max(1, len(chunk_strs) + 1)
        for i, piece in enumerate(chunk_strs, start=1):
            await _sleep_until(t0, tool.start_offset_ms + per_chunk * i)
            yield SimpleNamespace(
                type="content_block_delta",
                index=block_idx,
                delta=SimpleNamespace(type="input_json_delta", partial_json=piece),
            )

    await _sleep_until(t0, workload.stream_duration_ms)
    yield SimpleNamespace(type="message_stop")


async def _sleep_until(t0: float, target_ms: float) -> None:
    now = asyncio.get_event_loop().time()
    delay = (target_ms / 1000.0) - (now - t0)
    if delay > 0:
        await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


def _spread(count: int, total_ms: float, *, start_pad_ms: float = 200.0) -> list[float]:
    """`count` evenly-spaced offsets within [start_pad_ms, total_ms - start_pad_ms]."""
    if count == 1:
        return [start_pad_ms]
    span = max(0.0, total_ms - 2 * start_pad_ms)
    step = span / (count - 1) if count > 1 else 0.0
    return [start_pad_ms + i * step for i in range(count)]


def workload_3_analytics() -> Workload:
    """3-tool analytics query: 3 tools × 1s, ~3s stream."""
    delays = [1000.0, 1000.0, 1000.0]
    offsets = _spread(3, 3000.0)
    names = ["fetch_metric", "fetch_dimension", "fetch_segment"]
    return Workload(
        name="3-tool analytics",
        stream_duration_ms=3000.0,
        tools=[
            ToolSpec(name=n, delay_ms=d, start_offset_ms=o)
            for n, d, o in zip(names, delays, offsets, strict=True)
        ],
    )


def workload_8_audit() -> Workload:
    """8-tool cost audit: mixed 0.5–2s, ~6s stream."""
    delays = [500.0, 800.0, 1200.0, 1500.0, 700.0, 2000.0, 900.0, 1100.0]
    offsets = _spread(8, 6000.0)
    names = [f"audit_{i}" for i in range(8)]
    return Workload(
        name="8-tool cost audit",
        stream_duration_ms=6000.0,
        tools=[
            ToolSpec(name=n, delay_ms=d, start_offset_ms=o)
            for n, d, o in zip(names, delays, offsets, strict=True)
        ],
    )


def workload_15_security() -> Workload:
    """15-tool multi-account security sweep: mixed 0.3–4s, ~10s stream."""
    delays = [
        300.0,
        500.0,
        800.0,
        1200.0,
        1500.0,
        2000.0,
        2500.0,
        3000.0,
        3500.0,
        4000.0,
        700.0,
        900.0,
        1100.0,
        1300.0,
        1700.0,
    ]
    offsets = _spread(15, 10000.0)
    names = [f"scan_{i}" for i in range(15)]
    return Workload(
        name="15-tool security sweep",
        stream_duration_ms=10000.0,
        tools=[
            ToolSpec(name=n, delay_ms=d, start_offset_ms=o)
            for n, d, o in zip(names, delays, offsets, strict=True)
        ],
    )


WORKLOADS: dict[str, Workload] = {
    "3": workload_3_analytics(),
    "8": workload_8_audit(),
    "15": workload_15_security(),
}
