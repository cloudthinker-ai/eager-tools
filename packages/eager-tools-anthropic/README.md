# eager-tools-anthropic

Anthropic adapter for **eager tool calling** — wraps `anthropic.AsyncMessageStream` and dispatches each tool the instant its block seals, overlapping tool execution with ongoing LLM generation.

> For the mechanism and runtime contract, see the [top-level `METHOD.md`](../../METHOD.md).

## Status

**Alpha / scaffold.** Public API shape is locked; implementation bodies arrive in Move 3 (CloudThinker port). See [`NEXT.md`](../../NEXT.md).

## Install (once published)

```bash
pip install eager-tools-anthropic
```

## Usage

```python
import anthropic
from eager_tools_anthropic import AnthropicEagerStream

client = anthropic.AsyncAnthropic()

async with client.messages.stream(
    model="claude-opus-4-6",
    tools=[...],
    messages=[...],
) as source:
    runner = AnthropicEagerStream(source, tools=my_tools)

    async for event in runner.events():
        print(event.kind, event.tool_call)

    async for call, result in runner.results():
        print(call.name, "→", result)
```

## Supported SDK versions

- `anthropic>=0.39` (Messages streaming API).

## License

MIT. See top-level [`LICENSE`](../../LICENSE).
