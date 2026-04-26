# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingImports=false, reportMissingTypeStubs=false
"""07 — LangGraph live via OpenRouter: `create_agent` + EagerMiddleware.

Mirrors `06_langgraph_live.py` but routes through OpenRouter's OpenAI-compatible
endpoint, so any tool-capable model on https://openrouter.ai/models works.
The middleware is provider-agnostic — `langchain-openai` exposes the same
`AIMessageChunk.tool_call_chunks` shape that `langchain-anthropic` does.

Requires: OPENROUTER_API_KEY in env, plus `langchain`, `langgraph`,
`langchain-openai`.

Run:

    OPENROUTER_API_KEY=sk-or-v1-... python examples/07_langgraph_openrouter.py
    # or:  make example-7
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

T0 = time.perf_counter()


def _ts() -> str:
    return f"[{(time.perf_counter() - T0) * 1000:7.1f} ms]"


class _SlowTool:
    """Minimal eager-tools `Tool` protocol implementation."""

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
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY to run this example.")
        return

    try:
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        print(f"Install langchain + langchain-openai: {exc}")
        return

    from eager_tools_langgraph import eager_middleware

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

    model = ChatOpenAI(
        model=DEFAULT_MODEL,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=60.0,
        streaming=True,
    )

    # Pre-bind tools to the model. With `ChatOpenAI(base_url=openrouter, …)`,
    # `create_agent`'s implicit binding doesn't always reach the underlying
    # request — pre-binding makes the contract explicit and works with any
    # OpenAI-compatible gateway.
    bound_model = model.bind_tools([get_weather, get_stock_price, get_news])

    agent = create_agent(
        model=bound_model,
        tools=[get_weather, get_stock_price, get_news],
        middleware=[eager_middleware(eager_tools_dict)],
    )

    print(f"{_ts()} invoking agent  model={DEFAULT_MODEL} …")
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
        "Each tool dispatched the moment its JSON block sealed — well before\n"
        "the model finished streaming. Without `eager_middleware`, three 2s tools\n"
        "would still pay 2s after stream end."
    )


if __name__ == "__main__":
    asyncio.run(main())
