"""Public type contracts for eager-tools-core.

These protocols and dataclasses are the load-bearing API surface. Every adapter
(anthropic, openai, langgraph, claude-agent) consumes and emits these types —
getting the shape wrong here propagates forever.

Stability: the runtime itself may change freely until v1.0. These types must not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

SealKind = Literal["tool_sealed", "message_complete", "cancelled"]


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

    `kind == "tool_sealed"`     → `tool_call` populated; dispatch it.
    `kind == "message_complete"` → stream ended cleanly.
    `kind == "cancelled"`        → stream aborted; in-flight tools must cancel.
    """

    kind: SealKind
    tool_call: ToolCall | None = None
    seal_latency_ms: float | None = None


@runtime_checkable
class Tool(Protocol):
    """Contract every user-registered tool must satisfy.

    `idempotent=False` tools are routed to the classic (non-eager) path
    automatically — they only execute after message_stop.
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
