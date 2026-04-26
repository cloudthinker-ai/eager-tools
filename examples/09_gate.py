"""09 — Per-call gate: deny eager dispatch based on the parsed args.

Demonstrates the optional `gate` attribute on a tool. The gate runs after
the JSON block seals (so args are visible) and decides per-call whether to
fire eagerly. Denied calls surface in `results()` as a `GateDeniedError`;
the underlying tool is never invoked here.

Run:

    python examples/09_gate.py

Why a gate (vs just `idempotent=False`):

- `idempotent=False` is a per-tool blanket policy.
- `gate` is per-call with parsed args visible — `read_file(path="/etc/...")`
  can be denied while `read_file(path="/var/log/...")` proceeds.

Critical: gates run on the stream-consumption critical path. Keep them
sync-fast (in-memory predicate, cached policy lookup). Slow human approval
flows belong at the agent-framework layer (e.g. LangGraph `interrupt()`),
not in the gate.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from eager_tools import GateDeniedError, Tool, ToolCall
from eager_tools_anthropic import AnthropicEagerStream


class ReadFile:
    """Idempotent read with a path-allowlist gate. The gate denies any path
    rooted at `/etc/`. Sync-fast — in-memory string predicate.
    """

    name = "read_file"
    idempotent = True

    async def gate(self, call: ToolCall) -> bool:
        path = call.arguments.get("path", "")
        return not path.startswith("/etc/")

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        return f"<contents of {arguments['path']}>"


async def fake_stream() -> AsyncIterator[Any]:
    """Two read_file calls — one allowed, one denied by the gate."""

    def cb_start(idx: int, tid: str, name: str) -> SimpleNamespace:
        return SimpleNamespace(
            type="content_block_start",
            index=idx,
            content_block=SimpleNamespace(type="tool_use", id=tid, name=name),
        )

    def delta(idx: int, partial: str) -> SimpleNamespace:
        return SimpleNamespace(
            type="content_block_delta",
            index=idx,
            delta=SimpleNamespace(type="input_json_delta", partial_json=partial),
        )

    yield cb_start(0, "ALLOW", "read_file")
    yield delta(0, '{"path":"/var/log/app.log"}')
    yield cb_start(1, "DENY", "read_file")
    yield delta(1, '{"path":"/etc/shadow"}')
    yield SimpleNamespace(type="message_stop")


async def main() -> None:
    tools: dict[str, Tool] = {"read_file": ReadFile()}
    stream = AnthropicEagerStream(fake_stream(), tools=tools)

    async for ev in stream.events():
        if ev.kind == "tool_sealed" and ev.tool_call is not None:
            print(f"sealed   {ev.tool_call.tool_call_id}  args={ev.tool_call.arguments}")

    print()
    async for call, payload in stream.results():
        if isinstance(payload, GateDeniedError):
            print(f"DENIED   {call.tool_call_id}  ({payload})")
        elif isinstance(payload, BaseException):
            print(f"ERROR    {call.tool_call_id}  {type(payload).__name__}: {payload}")
        else:
            print(f"OK       {call.tool_call_id}  → {payload}")


if __name__ == "__main__":
    asyncio.run(main())
