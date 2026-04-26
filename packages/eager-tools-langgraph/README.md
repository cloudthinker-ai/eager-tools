# eager-tools-langgraph

> **LangGraph adapter for `eager-tools`** — a one-line `AgentMiddleware` that
> overlaps tool execution with model streaming inside `langchain.agents.create_agent`.

The middleware owns the model's `astream(...)` loop, watches `tool_call_chunks`
fly past, and dispatches each idempotent tool the moment its JSON block seals —
not after `message_stop`. Returns a `ModelResponse(result=[AIMessage, ToolMessage₁, …])`
so the agent commits the assistant message and eagerly-resolved tool results in
a single graph step.

See the parent repo [`README.md`](../../README.md) for the eager-dispatch
benchmark headline (1.20× – 1.50× over classic parallel dispatch across 16
workloads).

---

## Install

```bash
pip install eager-tools-core eager-tools-langgraph langchain-anthropic
# (or langchain-openai — the middleware is provider-agnostic)
```

## 60-second quickstart

```python
import asyncio
from eager_tools_langgraph import eager_middleware
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return open(path).read()


# eager-tools needs a Tool-protocol object: name, idempotent flag, async __call__.
class ReadFileEager:
    name = "read_file"
    idempotent = True  # safe to fire mid-stream
    async def __call__(self, args: dict) -> str:
        return open(args["path"]).read()


async def main() -> None:
    agent = create_agent(
        model=ChatAnthropic(model_name="claude-sonnet-4-5", timeout=60.0, stop=None),
        tools=[read_file],
        middleware=[eager_middleware({"read_file": ReadFileEager()})],
    )
    result = await agent.ainvoke({"messages": [HumanMessage("read README.md and summarize")]})
    print(result["messages"][-1].content)


asyncio.run(main())
```

Runnable variants:

- [`examples/06_langgraph_live.py`](../../examples/06_langgraph_live.py) — `make example-6`, `ANTHROPIC_API_KEY` + `langchain-anthropic`.
- [`examples/07_langgraph_openrouter.py`](../../examples/07_langgraph_openrouter.py) — `make example-7`, any tool-capable OpenRouter model via `langchain-openai`.
- [`examples/08_langgraph_compare.py`](../../examples/08_langgraph_compare.py) — `make example-8`, runs the same workload **sequential vs parallel vs eager** and prints a timing summary.

> **OpenAI-compatible gateways (OpenRouter, vLLM, etc.):** with
> `ChatOpenAI(base_url=…)`, pre-bind tools via `model.bind_tools([…])` before
> passing to `create_agent` — the agent's implicit binding doesn't always
> reach the underlying request. Examples 07 and 08 do this; example 06
> doesn't need to (`langchain-anthropic` binds correctly).

---

## Version matrix

| Package          | Tested floor   | Why                                                    |
|------------------|----------------|--------------------------------------------------------|
| `langgraph`      | `>=1.1.6`      | Middleware seam stable; subgraph stream_mode bug land. |
| `langchain`      | `>=1.0`        | `langchain.agents.create_agent` is the entry point.    |
| `langchain-core` | `>=1.2.14`     | PR #35281 fixed parallel `tool_call_chunks` merge.     |
| `python`         | `>=3.11`       | Matches `eager-tools-core`.                            |

`langgraph 1.1.6`'s own pyproject only floors `langchain-core>=1.0.0`, but a
real bug in tool_call_chunks merging was fixed at `1.2.14`. We pin tighter
ourselves so the middleware doesn't silently lose parallel calls.

## Provider coverage

The middleware is provider-agnostic: any `BaseChatModel` whose `astream(...)`
yields LangChain `AIMessageChunk` with `tool_call_chunks` works. That covers
`langchain-anthropic`, `langchain-openai`, and most community providers.

No per-provider chunk normalizer needed — LangChain has already done that
work.

---

## Honest limits

1. **`create_agent` only.** Raw `StateGraph` with a custom model node is out
   of scope this version — the `awrap_model_call` middleware seam is bound to
   `create_agent`. If there's demand, a `StateGraph`-friendly helper lands in
   v0.2.1.

2. **Per-agent middleware.** Subgraphs need the middleware re-registered on
   every `create_agent` instance. The framework does not auto-propagate it.

3. **`add_messages` ordering.** The middleware commits `[AIMessage, ToolMessage…]`
   together. If you have a custom message reducer that re-sorts by timestamp,
   ordering is undefined.

4. **Streamed token observability.** Because the middleware owns the
   `astream(...)` loop, downstream callers using `stream_mode="messages"`
   on the agent won't see token deltas from the wrapped node. Token-by-token
   UIs need to plumb chunks through `get_stream_writer()` as a `custom`
   event — open an issue if you need this baked in.

5. **Provider chunk shape variance.** The adapter trusts LangChain's normalized
   `tool_call_chunks` shape. Spec-compliant providers Just Work; for known
   upstream issues — GPT-5 content+tool_calls interleaving (langchain-ai/langchain#6510),
   Gemini parallel call drops (#10196) — the fix has to land upstream first.

---

## Running tests

```bash
make test-langgraph                  # 7 replay + 1 create_agent smoke test
```

## Design rationale

See [`~/.claude-duc/plans/plan-langgraph-adapter.md`](.) for the full design
walkthrough — why `awrap_model_call` over `ToolNode` subclassing, how the
seal/dispatch loop maps to `ModelResponse.result`, and the `KeyError: 'model'`
gotcha that requires the no-op `after_model` override.

For the underlying eager-dispatch mechanism (provider-agnostic), see the
top-level [`METHOD.md`](../../METHOD.md).
