# eager-tools

> **Cut agent wall-clock latency by overlapping tool execution with LLM streaming.**
>
> A production-grade reference implementation of **eager tool calling** — the pattern that dispatches each tool the moment its block finishes streaming, not after `message_stop`.

[![PyPI](https://img.shields.io/pypi/v/eager-tools-core.svg)](https://pypi.org/project/eager-tools-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![CI](https://github.com/cloudthinker-ai/eager-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/cloudthinker-ai/eager-tools/actions)
[![Docs](https://img.shields.io/badge/docs-METHOD.md-blue.svg)](./METHOD.md)

---

## The problem in one graph

<p align="center">
  <img src="./docs/diagrams/eager-vs-classic-timeline.svg?v=3" alt="Timeline: parallel waits for stream to finish; eager fires each tool the moment its block seals — tools and stream overlap."/>
</p>

<details>
<summary>ASCII fallback</summary>

```
Classic parallel tool calling:
stream : [==================================]
tools  :                                     [===========]  ← idle during stream
total  :                                                    ← stream + max(tool)

Eager tool calling:
stream : [==================================]
tool A :   [=========]        ← fires mid-stream
tool B :       [=========]    ← fires mid-stream, overlaps A
tool C :           [=========]← fires at message_stop
total  : [==================================]               ← max(stream, max(tool))
```

</details>

Parallel tool calling overlaps tools with tools. **Eager tool calling overlaps tools with generation itself.**

## Benchmark headline

Synthetic harness — `make bench` reproduces locally, deterministic.
Across 16 workloads (3 → 15 tools), eager beats parallel by **1.20× – 1.50×** (median ~1.28×).
Parallel is the right baseline: modern frameworks (`langchain.agents.create_agent`,
OpenAI Agents SDK, Vercel AI SDK) already execute tool calls from one
assistant message concurrently. Eager's win comes from overlapping tools
with the *stream itself* — something parallel dispatch can't do.
Full table + repro details: [`bench/results.md`](./bench/results.md).

<p align="center">
  <img src="./docs/diagrams/benchmark-results-chart.svg?v=3" alt="Bar chart: sequential vs parallel vs eager across 3-tool, 9-tool, and 15-tool workloads. Eager wins by 1.21×–1.46× vs parallel."/>
</p>

| Workload | Sequential | Parallel | **Eager** | Speedup vs parallel |
|----------|------------|----------|-----------|---------------------|
| 3-tool analytics | 4.90s | 3.50s | **2.90s** | 1.21× |
| 9-tool incident triage | 17.61s | 9.50s | **6.50s** | 1.46× |
| 15-tool ad campaign | 30.42s | 11.50s | **8.80s** | 1.31× |

> These are **lower bounds**. The synthetic stream removes network jitter,
> tail latency, and provider-side variance — the things that make eager
> dispatch shine in production. Run `make bench-live-anthropic` (or
> `-openai`) to spot-check against a real provider.

---

## 60-second quickstart

```bash
pip install eager-tools-core eager-tools-langgraph   # once published
# or, from source:
git clone https://github.com/cloudthinker-ai/eager-tools && cd eager-tools && make sync
```

```python
import asyncio, os
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from eager_tools_langgraph import eager_middleware

class SlowTool:
    def __init__(self, name: str, delay: float = 2.0):
        self.name = name
        self.idempotent = True
        self._delay = delay
    async def __call__(self, arguments):
        await asyncio.sleep(self._delay)
        return {"name": self.name, "args": arguments, "ok": True}

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

eager_tools = {
    "get_weather":    SlowTool("get_weather"),
    "get_stock_price": SlowTool("get_stock_price"),
    "get_news":       SlowTool("get_news"),
}

async def main():
    agent = create_agent(
        model=ChatAnthropic(model_name="claude-sonnet-4-5", timeout=60.0, stop=None),
        tools=[get_weather, get_stock_price, get_news],
        middleware=[eager_middleware(eager_tools)],
    )
    result = await agent.ainvoke({
        "messages": [HumanMessage(
            "Get the weather in NYC, the AAPL stock price, and recent AI news."
        )]
    })
    print(result["messages"][-1].content)

asyncio.run(main())
```

One middleware line wires eager dispatch into any `create_agent` call — no
changes to your tools or prompt. Works with OpenAI too: swap `ChatAnthropic`
for `ChatOpenAI`. Runnable variants in [`examples/`](./examples).

---

## Why this exists

Modern agent APIs — Anthropic, OpenAI, Bedrock — let the model emit multiple `tool_use` blocks in one assistant message and run them in parallel. That moves the tool phase from *sum* of durations to *max*. Good, but insufficient.

The **stream phase still happens first**. Tools still wait for `message_stop`. A four-second model stream followed by 2.5s of parallel tool execution is 6.5 seconds of wall clock. Eager tool calling makes it 4 seconds — the tools run *during* the stream, not after it.

See [`METHOD.md`](./METHOD.md) for the full mechanism: the seal event, the `tool_call_id` invariant, the runtime contract, and the edge cases.

<p align="center">
  <img src="./docs/diagrams/seal-mechanism-flow.svg?v=3" alt="Seal mechanism: a new tool_call_id in the stream triggers a SealEvent, which dispatches the completed tool to the ExecutorPool while the stream buffers the next block."/>
</p>

---

<p align="center">
  <img src="./docs/diagrams/stream-handler-architecture.svg?v=3" alt="Stream handler architecture: provider stream → adapter → SealDetector → ExecutorPool → user events and results."/>
</p>

For the per-block mechanism (chunks → buffer → seal → dispatch), see
[`docs/diagrams/seal-mechanism-flow.svg`](./docs/diagrams/seal-mechanism-flow.svg).

## When NOT to use it

- **Fast tools (sub-50ms).** Seal/dispatch overhead exceeds the latency saved.
- **Sequentially dependent tools.** If tool B needs tool A's result, the model won't emit B until A returns — no pipeline opportunity.
- **Non-idempotent tools.** Payments, destructive commands, outbound messages. Route these to the classic path via `Tool.idempotent = False` for blanket denial, or via a per-call `gate` callable for case-by-case decisions with parsed args visible (e.g. allow `read_file` but not under `/etc/`). See [`docs/hitl.md`](./docs/hitl.md). The gate still gates the *eager* path; the underlying tool still runs at the framework's tool step for non-denied calls.
- **Non-streaming backends.** If your gateway buffers the full response, eager dispatch is impossible.

Long version with edge cases: [`docs/when-not-to-use.md`](./docs/when-not-to-use.md).

## Contributing

Adapter PRs welcome — LlamaIndex, AutoGen, Vercel AI SDK, any provider that exposes a streaming response with per-block identifiers. Start from `packages/eager-tools-core/` as the contract reference. See [`NEXT.md`](./NEXT.md) §3 for the extraction pattern.

Bug reports + design discussions happen in **GitHub Discussions** — issues are intentionally disabled to keep the signal-to-noise ratio high.

## Acknowledgements

This pattern was extracted from production at [CloudThinker](https://cloudthinker.io), where it cuts median agent task latency by 50%. Internal codename: *tool-call pipelining*. External name: *eager tool calling*.

Read the full production story: [*Eager Tool Calling at CloudThinker*](https://cloudthinker.io/blogs/eager-tool-calling-21x-faster-agents).

## License

MIT — see [`LICENSE`](./LICENSE).
