"""Async executor pool for sealed tool calls.

The pool's lifetime is owned by the stream reader. When the stream dies
(user interrupt, model stop sequence, network error), `cancel_all()` releases
every in-flight tool task cleanly.

Non-idempotent tools are rejected by `dispatch()` — callers must route them to
the classic (non-eager) path that fires after message_stop.

See METHOD.md §4 for the full runtime contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from .types import NOOP_OBSERVABILITY, ObservabilityHook, Tool, ToolCall


class NonIdempotentToolError(RuntimeError):
    """Raised when a non-idempotent tool is offered to the eager executor.

    Non-idempotent tools (payments, destructive CLI commands, outbound messages)
    must NOT fire before `message_stop` — the model may retract them mid-stream.
    Callers should catch this and route to the classic dispatch path.
    """


class ExecutorPool:
    """Dispatches sealed tool calls on isolated asyncio tasks.

    Each task has its own exception boundary — one tool failing surfaces as an
    `is_error=True` result on the next turn; the pool and other tools keep running.
    """

    def __init__(
        self,
        tools: dict[str, Tool],
        *,
        observability: ObservabilityHook = NOOP_OBSERVABILITY,
        max_concurrent: int = 32,
    ) -> None:
        self._tools = tools
        self._observability = observability
        self._max_concurrent = max_concurrent

    async def dispatch(self, call: ToolCall) -> None:
        """Fire a sealed tool. Raises NonIdempotentToolError for unsafe tools.

        Returns immediately — the tool runs on a background task. Results arrive
        via `results()`.
        """
        _ = call
        raise NotImplementedError("Implementation deferred to Move 3 (port phase).")

    async def results(self) -> AsyncIterator[tuple[ToolCall, Any | Exception]]:
        """Async iterator over completed tool results, in completion order.

        Yields `(ToolCall, result)` on success, `(ToolCall, Exception)` on failure.
        Closes when the pool is cancelled or all dispatched tools finish after
        `close()` is awaited.
        """
        if False:  # pragma: no cover — keeps the signature an async generator
            yield  # type: ignore[unreachable]
        raise NotImplementedError("Implementation deferred to Move 3 (port phase).")

    async def cancel_all(self) -> None:
        """Cancel every in-flight task. Idempotent. Safe to call from any scope."""
        raise NotImplementedError("Implementation deferred to Move 3 (port phase).")

    async def close(self) -> None:
        """Signal no more dispatches will arrive. `results()` drains then closes."""
        raise NotImplementedError("Implementation deferred to Move 3 (port phase).")

    @property
    def in_flight(self) -> int:
        """Count of tool tasks currently running."""
        return 0


__all__ = ["ExecutorPool", "NonIdempotentToolError"]
