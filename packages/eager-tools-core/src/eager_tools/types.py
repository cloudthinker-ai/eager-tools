"""Public type contracts for eager-tools-core.

These protocols and dataclasses are the load-bearing API surface. Every adapter
(anthropic, openai, langgraph, claude-agent) consumes and emits these types —
getting the shape wrong here propagates forever.

Stability: the runtime itself may change freely until v1.0. These types must not.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

SealKind = Literal["tool_sealed", "message_complete", "cancelled"]


GateFn = Callable[["ToolCall"], Awaitable[bool]]
"""Per-call gate signature.

A gate inspects a sealed `ToolCall` and returns `True` to allow eager
dispatch or `False` to deny it. Returning `False` (or raising) routes the
call off the eager path — the adapter surfaces the denial in `results()`,
or the framework's tool step picks the call up.

Gates are awaited inside the same task that consumes the LLM stream; a slow
gate starves the stream and may trip provider keepalive. Keep gates
sync-fast (in-memory predicates, cached policy lookups). Slow approval
flows belong at the agent-framework layer (e.g. LangGraph `interrupt()`),
not in the gate.
"""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A fully-sealed tool call ready for dispatch.

    Emitted by SealDetector the instant a new tool_call_id arrives on the stream
    (sealing the previous tool) or the stream ends (sealing the final tool).
    """

    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    conversation_id: str | None = None


@dataclass(frozen=True, slots=True)
class SealEvent:
    """A seal boundary on the stream.

    `kind == "tool_sealed"`     → `tool_call` populated; dispatch it (unless
                                  `parse_error` is set, see below).
    `kind == "message_complete"` → stream ended cleanly.
    `kind == "cancelled"`        → stream aborted; in-flight tools must cancel.

    `parse_error` is set when the detector finished a tool block but couldn't
    materialize it (malformed JSON args, missing name). `tool_call` still
    carries the `tool_call_id` for downstream error reporting, but `arguments`
    is `{}` and the tool MUST NOT be dispatched. Adapters convert these into
    error tool messages so the model can recover on the next turn instead of
    the whole stream crashing.
    """

    kind: SealKind
    tool_call: ToolCall | None = None
    seal_latency_ms: float | None = None
    parse_error: BaseException | None = None


@runtime_checkable
class Tool(Protocol):
    """Contract every user-registered tool must satisfy.

    `idempotent=False` tools are routed to the classic (non-eager) path
    automatically — they only execute after message_stop.

    Optional per-call gate (duck-typed, not declared on this Protocol so adding
    one to an existing tool stays backward-compatible):

        class ReadFile:
            name = "read_file"
            idempotent = True

            async def gate(self, call: ToolCall) -> bool:
                return not call.arguments["path"].startswith("/etc/")

            async def __call__(self, arguments): ...

    The runtime reads `getattr(tool, "gate", None)` after the idempotency check.
    See `GateFn` for the signature contract and the critical-path warning.
    """

    name: str
    idempotent: bool

    async def __call__(self, arguments: dict[str, Any]) -> Any: ...


@runtime_checkable
class ObservabilityHook(Protocol):
    """Opt-in trace emitter. Runtime calls these on every lifecycle event.

    Default implementation is a no-op. Integrators plug in OTel / Langfuse /
    LangSmith by implementing this protocol.
    """

    def on_seal(self, event: SealEvent) -> None: ...

    def on_dispatch_start(self, call: ToolCall) -> None: ...

    def on_dispatch_end(self, call: ToolCall, error: Exception | None) -> None: ...


class _NoopObservability:
    """Default hook. Silent. Zero overhead."""

    def on_seal(self, event: SealEvent) -> None:
        del event

    def on_dispatch_start(self, call: ToolCall) -> None:
        del call

    def on_dispatch_end(self, call: ToolCall, error: Exception | None) -> None:
        del call, error


NOOP_OBSERVABILITY: ObservabilityHook = _NoopObservability()
