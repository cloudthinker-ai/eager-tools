"""04 — Cancellation: tear down a stream mid-flight cleanly.

Demonstrates that cancelling the events() task releases all in-flight tool
tasks via the executor pool's cancellation scope. No "Task was destroyed"
warnings, no leaked tasks.

Run:

    python examples/04_cancellation.py
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from eager_tools import Tool
from eager_tools_anthropic import AnthropicEagerStream

T0 = time.perf_counter()


def _ts() -> str:
    return f"[{(time.perf_counter() - T0) * 1000:6.1f} ms]"


class LongRunningTool:
    """Idempotent fake tool that takes 5s — way longer than the user's patience."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.idempotent = True

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        del arguments
        print(f"{_ts()}   ↳ {self.name} started (will run 5s)")
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            print(f"{_ts()}   ↳ {self.name} CANCELLED cleanly")
            raise
        print(f"{_ts()}   ↳ {self.name} done")
        return {"name": self.name}


async def slow_provider() -> AsyncIterator[Any]:
    """A stream that emits 3 tool blocks then pauses forever."""

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

    yield cb_start(0, "A", "slow")
    yield delta(0, "{}")
    yield cb_start(1, "B", "slow")  # seals A → A starts running
    yield delta(1, "{}")
    yield cb_start(2, "C", "slow")  # seals B → B starts running
    yield delta(2, "{}")
    # No message_stop — model hangs, user gets impatient.
    await asyncio.sleep(60)


async def main() -> None:
    tools: dict[str, Tool] = {"slow": LongRunningTool("slow")}
    stream = AnthropicEagerStream(slow_provider(), tools=tools)

    async def drain_events() -> None:
        async for ev in stream.events():
            if ev.kind == "tool_sealed" and ev.tool_call is not None:
                print(f"{_ts()} sealed {ev.tool_call.tool_call_id} — dispatched")

    task = asyncio.create_task(drain_events())

    # Give the stream 1s to dispatch some tools, then bail.
    await asyncio.sleep(1.0)
    print(f"{_ts()} user cancels — calling stream.cancel()")
    await stream.cancel()

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    print(f"{_ts()} all tool tasks released. clean exit.")


if __name__ == "__main__":
    asyncio.run(main())
