# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingImports=false
"""02 — Anthropic live: real Claude stream, three idempotent fake tools.

Requires: ANTHROPIC_API_KEY in env, `pip install anthropic`.
Run:

    ANTHROPIC_API_KEY=... python examples/02_anthropic_live.py

You should see each tool dispatch the moment its JSON block seals, well
before Claude finishes streaming — wall clock ≈ slowest tool, not sum.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from eager_tools import Tool
from eager_tools_anthropic import AnthropicEagerStream

T0 = time.perf_counter()


def _ts() -> str:
    return f"[{(time.perf_counter() - T0) * 1000:7.1f} ms]"


class SlowTool:
    def __init__(self, name: str, delay: float = 1.0) -> None:
        self.name = name
        self.idempotent = True
        self._delay = delay

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        print(f"{_ts()}   ↳ {self.name} started   args={arguments}")
        await asyncio.sleep(self._delay)
        print(f"{_ts()}   ↳ {self.name} done")
        return {"name": self.name, "args": arguments, "ok": True}


TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_news",
        "description": "Get recent news on a topic.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
]


async def main() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Set ANTHROPIC_API_KEY to run this example.")
        return

    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        print("Install anthropic: `pip install anthropic`")
        return

    client = AsyncAnthropic()
    tools: dict[str, Tool] = {
        "get_weather": SlowTool("get_weather"),
        "get_stock_price": SlowTool("get_stock_price"),
        "get_news": SlowTool("get_news"),
    }

    print(f"{_ts()} requesting Claude stream …")
    async with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        tools=TOOL_SCHEMAS,  # type: ignore[arg-type]
        messages=[
            {
                "role": "user",
                "content": (
                    "Get the weather in NYC, the AAPL stock price, and recent AI news. "
                    "Call all three tools in parallel."
                ),
            }
        ],
    ) as raw_stream:
        stream = AnthropicEagerStream(raw_stream, tools=tools)
        async for ev in stream.events():
            if ev.kind == "tool_sealed" and ev.tool_call is not None:
                print(
                    f"{_ts()} sealed {ev.tool_call.name:18s}  "
                    f"latency={ev.seal_latency_ms:6.1f} ms — dispatched eagerly"
                )
            elif ev.kind == "message_complete":
                print(f"{_ts()} message_complete (tools may still be running)")

        async for call, result in stream.results():
            print(f"{_ts()} result {call.name:18s} ← {result}")

    elapsed = (time.perf_counter() - T0) * 1000
    print(
        f"\n{_ts()} done. Total: {elapsed:.0f} ms.\n"
        "Compare to classic dispatch: 3 sequential 1s tools after stream → ~3000 ms more."
    )


if __name__ == "__main__":
    asyncio.run(main())
