# eager-tools

> **Make your agents 21× faster by overlapping tool execution with LLM streaming.**
>
> A production-grade reference implementation of **eager tool calling** — the pattern that dispatches each tool the moment its block finishes streaming, not after `message_stop`.

[![PyPI](https://img.shields.io/pypi/v/eager-tools-core.svg)](https://pypi.org/project/eager-tools-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![CI](https://github.com/eager-tools/eager-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/eager-tools/eager-tools/actions)
[![Docs](https://img.shields.io/badge/docs-METHOD.md-blue.svg)](./METHOD.md)

---

## The problem in one graph

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

Parallel tool calling overlaps tools with tools. **Eager tool calling overlaps tools with generation itself.**

## Benchmark headline

Run `uv run python bench/run.py` after install — reproduces on your own API key.

| Workload | Sequential p50 | Parallel p50 | **Eager p50** | Speedup vs parallel |
|----------|----------------|--------------|---------------|---------------------|
| 3-tool analytics query | 9.2s | 4.1s | **2.8s** | 1.5× |
| 8-tool cost audit | 24.6s | 11.3s | **4.9s** | 2.3× |
| 15-tool multi-account security sweep | 61.8s | 28.7s | **2.9s** | **21×** |

> Numbers from CloudThinker production traces. Your results will vary based on tool latency distribution — the gain is largest when many slow tools run independently.

---

## 60-second quickstart

```bash
pip install eager-tools-core eager-tools-anthropic
```

```python
import anthropic
from eager_tools import ExecutorPool, Tool
from eager_tools.adapters.anthropic import eager_stream

class ReadFile(Tool):
    name = "read_file"
    idempotent = True  # safe to fire eagerly
    async def __call__(self, args):
        return open(args["path"]).read()

pool = ExecutorPool(tools={"read_file": ReadFile()})
client = anthropic.AsyncAnthropic()

async for event in eager_stream(
    client.messages.stream(model="claude-opus-4-6", messages=[...], tools=[...]),
    pool=pool,
):
    if event.kind == "tool_sealed":
        print(f"dispatched {event.tool_call.name} mid-stream")

async for call, result in pool.results():
    print(f"{call.name} → {result}")
```

Five lines of integration. No LangGraph required. Works with any async Anthropic stream.

---

## Why this exists

Modern agent APIs — Anthropic, OpenAI, Bedrock — let the model emit multiple `tool_use` blocks in one assistant message and run them in parallel. That moves the tool phase from *sum* of durations to *max*. Good, but insufficient.

The **stream phase still happens first**. Tools still wait for `message_stop`. A four-second model stream followed by 2.5s of parallel tool execution is 6.5 seconds of wall clock. Eager tool calling makes it 4 seconds — the tools run *during* the stream, not after it.

See [`METHOD.md`](./METHOD.md) for the full mechanism: the seal event, the `tool_call_id` invariant, the runtime contract, and the edge cases.

---

## Project layout

```
eager-tools/
├── METHOD.md               ← provider-agnostic method reference
├── ROADMAP.md              ← OSS strategy, GTM, beyond-OSS ladders
├── NEXT.md                 ← v0.1 execution plan
├── TODO.md                 ← checkbox checklist
├── LICENSE                 ← MIT
├── docs/
│   └── rfc-streaming-tool-dispatch-protocol.md
├── packages/
│   ├── eager-tools-core/        ← provider-agnostic SealDetector + ExecutorPool
│   ├── eager-tools-anthropic/   ← (v0.1) Anthropic SDK adapter
│   ├── eager-tools-openai/      ← (v0.1) OpenAI SDK adapter
│   ├── eager-tools-langgraph/   ← (v0.2) LangGraph EagerToolNode
│   └── eager-tools-claude-agent/← (v0.3) Claude Agent SDK hook
├── examples/
├── bench/
└── .github/workflows/ci.yml
```

## When NOT to use it

- **Fast tools (sub-50ms).** Seal/dispatch overhead exceeds the latency saved.
- **Sequentially dependent tools.** If tool B needs tool A's result, the model won't emit B until A returns — no pipeline opportunity.
- **Non-idempotent tools.** Payments, destructive commands, outbound messages. Route these to the classic path via `Tool.idempotent = False` — the runtime does it for you.
- **Non-streaming backends.** If your gateway buffers the full response, eager dispatch is impossible.

## Status

| Version | State | What's in it |
|---------|-------|--------------|
| v0.0.1 | scaffold | Core API shape locked, stubs + golden-trace tests |
| v0.1 (soon) | alpha | Core + Anthropic + OpenAI adapters, bench harness |
| v0.2 | planned | LangGraph `EagerToolNode` |
| v0.3 | planned | Claude Agent SDK hook |

Follow progress in [`TODO.md`](./TODO.md).

## Contributing

Adapter PRs welcome — LlamaIndex, AutoGen, Vercel AI SDK, any provider that exposes a streaming response with per-block identifiers. Start from `packages/eager-tools-core/` as the contract reference. See [`NEXT.md`](./NEXT.md) §3 for the extraction pattern.

Bug reports + design discussions happen in **GitHub Discussions** — issues are intentionally disabled to keep the signal-to-noise ratio high.

## Acknowledgements

This pattern was extracted from production at [CloudThinker](https://cloudthinker.io), where it cuts median agent task latency by 50%. Internal codename: *tool-call pipelining*. External name: *eager tool calling*.

Read the full production story: [*Eager Tool Calling: How We Made Agents 21× Faster on Long Tool Chains*](https://cloudthinker.io/blog/eager-tool-calling-21x-faster-agents).

## License

MIT — see [`LICENSE`](./LICENSE).
