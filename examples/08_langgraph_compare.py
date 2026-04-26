# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingImports=false, reportMissingTypeStubs=false
"""08 — LangGraph live: sequential vs parallel vs eager dispatch, side by side.

Runs the same prompt + same three slow tools against the same model three ways.
The model always emits all three tool_calls in one assistant message; the
modes differ only in how the *runtime* dispatches them.

    sequential — runtime executes the model's tool_calls one-at-a-time, in
                 order. Mirrors a pre-parallel framework. Wall-clock =
                 stream + sum(tool).

    parallel   — runtime fans tool_calls out concurrently AFTER message_stop.
                 What modern `create_agent` / OpenAI Agents SDK / Vercel AI SDK
                 do by default. Wall-clock = stream + max(tool).

    eager      — runtime fires each tool the moment its JSON block seals,
                 overlapping with the rest of the stream. The eager-tools win.
                 Wall-clock = max(stream, max(tool_end)).

Sequential is hand-rolled (no `create_agent`) so the comparison is deterministic
across providers — `bind_tools(parallel_tool_calls=False)` is honored
unevenly across models and OpenAI-compatible gateways.

Requires: OPENROUTER_API_KEY in env, plus `langchain`, `langgraph`,
`langchain-openai`.

Run:

    OPENROUTER_API_KEY=sk-or-v1-... python examples/08_langgraph_compare.py
    # or:  make example-8

Heads-up: this issues 3 separate API calls (one per mode). Costs cents, not
dollars, against a small model. LLMs being non-deterministic, exact wall times
vary run-to-run; the *ordering* (sequential > parallel > eager on long tools)
is what's pedagogically real.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
TOOL_DELAY_S = 2.0
PROMPT = (
    "Get the weather in NYC, the AAPL stock price, and recent AI news. "
    "Call each of the three tools exactly once, then summarize in one final "
    "message. Do not call any tool more than once."
)


@dataclass
class _Run:
    label: str
    elapsed_ms: float
    tool_count: int
    message_count: int


class _SlowTool:
    """eager-tools Tool protocol — name, idempotent, async __call__."""

    def __init__(self, name: str, delay: float = TOOL_DELAY_S) -> None:
        self.name = name
        self.idempotent = True
        self._delay = delay
        self.fired_at_ms: list[float] = []

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        self.fired_at_ms.append(_now_ms())
        await asyncio.sleep(self._delay)
        return {"name": self.name, "args": arguments, "ok": True}


def _make_tools(label: str) -> tuple[list[Any], dict[str, _SlowTool]]:
    """Build the (langchain @tool wrappers, eager-tools dict) pair.

    Each call returns fresh instances so per-mode timing is clean.
    """
    from langchain_core.tools import tool

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

    eager_dict: dict[str, _SlowTool] = {
        "get_weather": _SlowTool("get_weather"),
        "get_stock_price": _SlowTool("get_stock_price"),
        "get_news": _SlowTool("get_news"),
    }
    # The langchain @tool stubs run for `parallel` and `sequential` modes
    # (LangGraph's ToolNode invokes them). For `eager`, the middleware
    # dispatches the eager-tools dict instead, so the @tool stubs return ""
    # and never matter.
    if label in ("parallel", "sequential"):

        @tool
        def get_weather_slow(city: str) -> str:  # type: ignore[no-redef]
            """Get current weather for a city."""
            return _slow_blocking("get_weather", "city", city)

        @tool
        def get_stock_price_slow(ticker: str) -> str:  # type: ignore[no-redef]
            """Get the current stock price for a ticker symbol."""
            return _slow_blocking("get_stock_price", "ticker", ticker)

        @tool
        def get_news_slow(topic: str) -> str:  # type: ignore[no-redef]
            """Get recent news on a topic."""
            return _slow_blocking("get_news", "topic", topic)

        # Override langchain wrappers to actually sleep, so wall-clock reflects
        # real tool latency in non-eager modes too.
        return (
            [get_weather_slow, get_stock_price_slow, get_news_slow],
            eager_dict,
        )
    return ([get_weather, get_stock_price, get_news], eager_dict)


def _slow_blocking(name: str, arg_name: str, arg_value: str) -> str:
    """Sync sleep so LangGraph ToolNode timings include real tool latency.

    Returns the same JSON shape `_SlowTool` does, so the model gets equally
    rich results across modes and is less tempted to re-call.
    """
    time.sleep(TOOL_DELAY_S)
    return (
        f'{{"name": "{name}", "args": {{"{arg_name}": "{arg_value}"}}, "ok": true}}'
    )


def _now_ms() -> float:
    return (time.perf_counter() - T0) * 1000


T0 = time.perf_counter()


def _build_model(api_key: str) -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=DEFAULT_MODEL,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=60.0,
        streaming=True,
    )


async def _run_via_agent(
    label: str,
    *,
    api_key: str,
    use_eager: bool,
) -> _Run:
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage, ToolMessage

    from eager_tools_langgraph import eager_middleware

    lc_tools, eager_dict = _make_tools(label)
    bound_model = _build_model(api_key).bind_tools(lc_tools)
    middleware = [eager_middleware(eager_dict)] if use_eager else []
    agent = create_agent(
        model=bound_model,
        tools=lc_tools,
        middleware=middleware,
    )

    started_ms = _now_ms()
    print(f"\n=== {label} === start @ {started_ms:7.1f} ms")
    result = await agent.ainvoke({"messages": [HumanMessage(PROMPT)]})
    elapsed_ms = _now_ms() - started_ms

    tool_count = sum(1 for m in result["messages"] if isinstance(m, ToolMessage))
    print(
        f"=== {label} === done  @ {_now_ms():7.1f} ms  "
        f"elapsed={elapsed_ms:7.0f} ms  tools={tool_count}  "
        f"messages={len(result['messages'])}"
    )
    return _Run(
        label=label,
        elapsed_ms=elapsed_ms,
        tool_count=tool_count,
        message_count=len(result["messages"]),
    )


async def _run_sequential(api_key: str) -> _Run:
    """Hand-rolled sequential dispatch: one round-trip, tools run serially.

    The model still emits all three tool_calls in one assistant message —
    that's a model behavior, not a runtime behavior. We then dispatch them
    one at a time, mirroring how a pre-parallel framework would have run
    them. Single round-trip, no looping, deterministic across providers.
    """
    from langchain_core.messages import HumanMessage

    lc_tools, _ = _make_tools("sequential")
    bound_model = _build_model(api_key).bind_tools(lc_tools)
    tools_by_name = {t.name: t for t in lc_tools}

    started_ms = _now_ms()
    print(f"\n=== sequential === start @ {started_ms:7.1f} ms")

    ai_msg = await bound_model.ainvoke([HumanMessage(PROMPT)])
    tool_calls = getattr(ai_msg, "tool_calls", []) or []
    if not tool_calls:
        print("  model returned no tool_calls — sequential demo aborted")
        elapsed_ms = _now_ms() - started_ms
        return _Run(label="sequential", elapsed_ms=elapsed_ms, tool_count=0, message_count=1)

    for tc in tool_calls:
        # ainvoke runs the @tool wrapper (sync `_slow_blocking`) in a thread —
        # awaiting it serializes the 2s sleeps end-to-end.
        await tools_by_name[tc["name"]].ainvoke(tc["args"])

    elapsed_ms = _now_ms() - started_ms
    print(
        f"=== sequential === done  @ {_now_ms():7.1f} ms  "
        f"elapsed={elapsed_ms:7.0f} ms  tools={len(tool_calls)}  "
        f"(hand-rolled — single model call + serial tool dispatch)"
    )
    return _Run(
        label="sequential",
        elapsed_ms=elapsed_ms,
        tool_count=len(tool_calls),
        message_count=2 + len(tool_calls),  # Human + AI(tool_calls) + N ToolMessages
    )


async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY to run this example.")
        return

    try:
        from langchain.agents import create_agent  # noqa: F401
        from langchain_openai import ChatOpenAI  # noqa: F401
    except ImportError as exc:
        print(f"Install langchain + langchain-openai: {exc}")
        return

    print(f"model={DEFAULT_MODEL}  tool_delay={TOOL_DELAY_S}s  prompt={PROMPT!r}")

    runs: list[_Run] = []
    runs.append(await _run_sequential(api_key=api_key))
    runs.append(await _run_via_agent("parallel", api_key=api_key, use_eager=False))
    runs.append(await _run_via_agent("eager", api_key=api_key, use_eager=True))

    parallel_ms = next(r.elapsed_ms for r in runs if r.label == "parallel")
    print("\n" + "=" * 60)
    print(f"{'mode':<12}  {'elapsed':>9}  {'tools':>5}  {'msgs':>4}  {'vs parallel':>11}")
    print("-" * 60)
    for r in runs:
        speedup = parallel_ms / r.elapsed_ms if r.elapsed_ms else float("nan")
        print(
            f"{r.label:<12}  {r.elapsed_ms:>7.0f}ms  {r.tool_count:>5}  "
            f"{r.message_count:>4}  {speedup:>10.2f}x"
        )
    print("=" * 60)
    print(
        "\nSequential pays stream + sum(tool). Parallel pays stream + max(tool).\n"
        "Eager pays max(stream, max(tool_end)) — the tools overlap the stream\n"
        "itself, not just each other."
    )


if __name__ == "__main__":
    asyncio.run(main())
