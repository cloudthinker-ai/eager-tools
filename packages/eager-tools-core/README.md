# eager-tools-core

Provider-agnostic core for **eager tool calling** — dispatches each tool the moment its block finishes streaming, overlapping tool execution with LLM generation on the wall clock.

> For the mechanism, runtime contract, and edge cases, see the [top-level `METHOD.md`](../../METHOD.md).
>
> For execution plan + OSS strategy, see [`ROADMAP.md`](../../ROADMAP.md) and [`NEXT.md`](../../NEXT.md).

## Status

**Alpha / scaffold.** Public API shape is locked; implementation bodies arrive in Move 3 (CloudThinker port). See [`NEXT.md`](../../NEXT.md) for the sequence.

## Install (once published)

```bash
pip install eager-tools-core
```

## Public API

```python
from eager_tools import (
    SealDetector,         # streaming chunk → SealEvent state machine
    ExecutorPool,         # async dispatcher for sealed tool calls
    ToolCall, SealEvent,  # data carriers
    Tool,                 # protocol every user tool implements
    ObservabilityHook,    # opt-in trace emitter (OTel / Langfuse / LangSmith)
    NonIdempotentToolError,
)
```

## Adapter contract

Each provider-specific adapter (anthropic, openai, langgraph, claude-agent) consumes `SealDetector` + `ExecutorPool` and emits normalized chunks into the detector. Adapters are separate packages — see `packages/eager-tools-<provider>/`.

## License

MIT. See top-level [`LICENSE`](../../LICENSE).
