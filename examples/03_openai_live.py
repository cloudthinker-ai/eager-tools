# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingImports=false
"""03 — OpenAI live: real GPT stream, three idempotent fake tools.

Requires: OPENAI_API_KEY in env, `pip install openai`.
Run:

    OPENAI_API_KEY=... python examples/03_openai_live.py

Same harness as the Anthropic example so you can compare side-by-side.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from eager_tools import Tool
from eager_tools_openai import OpenAIEagerStream

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


def _tool_schema(name: str, prop: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Get {name.replace('_', ' ')}.",
            "parameters": {
                "type": "object",
                "properties": {prop: {"type": "string"}},
                "required": [prop],
            },
        },
    }


TOOL_SCHEMAS = [
    _tool_schema("get_weather", "city"),
    _tool_schema("get_stock_price", "ticker"),
    _tool_schema("get_news", "topic"),
]


async def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        print("Set OPENAI_API_KEY to run this example.")
        return

    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("Install openai: `pip install openai`")
        return

    client = AsyncOpenAI()
    tools: dict[str, Tool] = {
        "get_weather": SlowTool("get_weather"),
        "get_stock_price": SlowTool("get_stock_price"),
        "get_news": SlowTool("get_news"),
    }

    print(f"{_ts()} requesting GPT stream …")
    raw_stream = await client.chat.completions.create(
        model="gpt-4o",
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
        stream=True,
    )

    stream = OpenAIEagerStream(raw_stream, tools=tools)
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
