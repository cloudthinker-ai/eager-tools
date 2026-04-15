# eager-tools-openai

OpenAI adapter for **eager tool calling** — wraps `chat.completions.create(stream=True)` and dispatches each tool the instant its block seals, overlapping tool execution with ongoing LLM generation.

> For the mechanism and runtime contract, see the [top-level `METHOD.md`](../../METHOD.md).

## Status

**Alpha / scaffold.** Public API shape is locked; implementation bodies arrive in Move 3 (CloudThinker port). See [`NEXT.md`](../../NEXT.md).

## Install (once published)

```bash
pip install eager-tools-openai
```

## Usage

```python
from openai import AsyncOpenAI
from eager_tools_openai import OpenAIEagerStream

client = AsyncOpenAI()

source = await client.chat.completions.create(
    model="gpt-4.1",
    tools=[...],
    messages=[...],
    stream=True,
)
runner = OpenAIEagerStream(source, tools=my_tools)

async for event in runner.events():
    print(event.kind, event.tool_call)

async for call, result in runner.results():
    print(call.name, "→", result)
```

## Supported SDK surfaces

- `chat.completions.create(stream=True)` with `tools=[...]` — **v0.1 target**.
- Responses API (`responses.stream(...)`) — **deferred**; contributions welcome.

## Supported SDK versions

- `openai>=1.50`.

## License

MIT. See top-level [`LICENSE`](../../LICENSE).
