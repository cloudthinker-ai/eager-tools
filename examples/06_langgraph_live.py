# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingImports=false, reportMissingTypeStubs=false
"""06 — LangGraph live: `create_agent` + EagerMiddleware + 3 fake slow tools.

Requires: ANTHROPIC_API_KEY in env, plus `langchain`, `langgraph`,
`langchain-anthropic`. The middleware seam works the same way for OpenAI —
swap `ChatAnthropic` for `ChatOpenAI` and set OPENAI_API_KEY.

Run:

    ANTHROPIC_API_KEY=... python examples/06_langgraph_live.py

You should see each tool dispatch the moment its JSON block seals (the
`MIDDLEWARE` lines), well before the model finishes streaming. Compare the
elapsed wall clock to the same agent without `eager_middleware` — three 2s
tools that classic-dispatch in parallel still pay 2s after stream end.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

T0 = time.perf_counter()


def _ts() -> str:
    return f"[{(time.perf_counter() - T0) * 1000:7.1f} ms]"


class _SlowTool:
    """Minimal eager-tools `Tool` protocol implementation: name, idempotent, __call__."""

    def __init__(self, name: str, delay: float = 2.0) -> None:
        self.name = name
        self.idempotent = True
        self._delay = delay

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        print(f"{_ts()}   ↳ {self.name} started   args={arguments}")
        await asyncio.sleep(self._delay)
        print(f"{_ts()}   ↳ {self.name} done")
        return {"name": self.name, "args": arguments, "ok": True}


async def main() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Set ANTHROPIC_API_KEY to run this example.")
        return

    try:
        from langchain.agents import create_agent
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
    except ImportError as exc:
        print(f"Install langchain + langchain-anthropic: {exc}")
        return

    from eager_tools_langgraph import eager_middleware

    # LangChain tool decorators — these are what the model "sees" as bindable
    # tools (schema, description). The eager runtime separately dispatches our
    # `_SlowTool` instances by name, so the actual implementations differ
    # slightly: we don't run langchain's `add.invoke(...)`, we run the eager
    # tool's `__call__`. They share the name and JSON shape only.
    @tool
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        return ""

    @tool
    def get_stock_price(ticker: str) -> str:
        """Get the current stock price for a ticker symbol."""
        return ""

    @tool
    def get_news(topic: str) -> str:
        """Get recent news on a topic."""
        return ""

    eager_tools_dict = {
        "get_weather": _SlowTool("get_weather"),
        "get_stock_price": _SlowTool("get_stock_price"),
        "get_news": _SlowTool("get_news"),
    }

    agent = create_agent(
        model=ChatAnthropic(model_name="claude-sonnet-4-5", timeout=60.0, stop=None),
        tools=[get_weather, get_stock_price, get_news],
        middleware=[eager_middleware(eager_tools_dict)],
    )

    print(f"{_ts()} invoking agent …")
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    "Get the weather in NYC, the AAPL stock price, and recent AI news. "
                    "Call all three tools in parallel."
                )
            ]
        }
    )

    elapsed = (time.perf_counter() - T0) * 1000
    print(f"\n{_ts()} done. Total: {elapsed:.0f} ms.")
    print(f"{_ts()} message count: {len(result['messages'])}")
    print(
        "Compare to classic dispatch (no middleware): three 2s tools fire AFTER\n"
        "the model stream ends → ~2000 ms additional wall clock."
    )


if __name__ == "__main__":
    asyncio.run(main())
