"""01 — Minimal: eager dispatch with a fake provider stream.

No SDK, no API key. Demonstrates the public surface end-to-end against an
in-memory event source. Run:

    python examples/01_minimal.py

What you'll see: tools start the moment their JSON block seals — well before
the model finishes streaming the rest of the message.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from eager_tools import Tool
from eager_tools_anthropic import AnthropicEagerStream

T0 = time.perf_counter()


def _ts() -> str:
    return f"[{(time.perf_counter() - T0) * 1000:6.1f} ms]"


class SlowTool:
    """Idempotent fake tool that sleeps `delay`s before returning."""

    def __init__(self, name: str, delay: float) -> None:
        self.name = name
        self.idempotent = True
        self._delay = delay

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        print(f"{_ts()}   ↳ {self.name} started")
        await asyncio.sleep(self._delay)
        print(f"{_ts()}   ↳ {self.name} done")
        return {"name": self.name, "args": arguments}


async def fake_anthropic_stream() -> AsyncIterator[Any]:
    """Three tool blocks streamed slowly, mimicking real provider chunk timing."""

    def cb_start(idx: int, tid: str, name: str) -> SimpleNamespace:
        return SimpleNamespace(
            type="content_block_start",
            index=idx,
            content_block=SimpleNamespace(type="tool_use", id=tid, name=name),
        )

    def delta(idx: int, partial: str) -> SimpleNamespace:
        return SimpleNamespace(
            type="content_block_delta",
            index=idx,
            delta=SimpleNamespace(type="input_json_delta", partial_json=partial),
        )

    # Tool A
    yield cb_start(0, "A", "fetch_weather")
    await asyncio.sleep(0.10)
    yield delta(0, '{"city":"NYC"}')

    # Tool B (sealing A immediately)
    yield cb_start(1, "B", "fetch_stock")
    await asyncio.sleep(0.10)
    yield delta(1, '{"ticker":"AAPL"}')

    # Tool C (sealing B)
    yield cb_start(2, "C", "fetch_news")
    await asyncio.sleep(0.10)
    yield delta(2, '{"topic":"ai"}')

    # Model continues streaming text for a while before finishing.
    await asyncio.sleep(0.30)
    yield SimpleNamespace(type="message_stop")


async def main() -> None:
    tools: dict[str, Tool] = {
        "fetch_weather": SlowTool("fetch_weather", delay=0.5),
        "fetch_stock": SlowTool("fetch_stock", delay=0.5),
        "fetch_news": SlowTool("fetch_news", delay=0.5),
    }
    stream = AnthropicEagerStream(fake_anthropic_stream(), tools=tools)

    print(f"{_ts()} stream opened — eager dispatch active")
    async for ev in stream.events():
        if ev.kind == "tool_sealed" and ev.tool_call is not None:
            print(
                f"{_ts()} sealed {ev.tool_call.name:14s}  "
                f"latency={ev.seal_latency_ms:6.1f} ms — dispatched eagerly"
            )
        elif ev.kind == "message_complete":
            print(f"{_ts()} message complete (tools may still be running)")

    async for call, result in stream.results():
        print(f"{_ts()} result {call.name:14s}  ← {result}")

    print(f"\n{_ts()} done. Without eager dispatch this would have taken ~3.0s+")


if __name__ == "__main__":
    asyncio.run(main())
