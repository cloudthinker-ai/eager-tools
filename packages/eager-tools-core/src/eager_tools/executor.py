"""Async executor pool for sealed tool calls.

The pool's lifetime is owned by the stream reader. When the stream dies
(user interrupt, model stop sequence, network error), `cancel_all()` releases
every in-flight tool task cleanly.

`dispatch()` denies eager execution under two conditions, both surfaced as
subclasses of `EagerDispatchDeniedError`:

- `NonIdempotentToolError` — `tool.idempotent is False` (per-tool blanket deny).
- `GateDeniedError`        — `tool.gate(call)` returned False or raised
                             (per-call deny with parsed args visible).

Adapters catch `EagerDispatchDeniedError` and choose how to surface it (record_error,
silent fall-through to the framework's tool step, etc.). The pool itself never
records on a denied dispatch.

See METHOD.md §4 for the full runtime contract.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from .types import NOOP_OBSERVABILITY, ObservabilityHook, Tool, ToolCall


class EagerDispatchDeniedError(RuntimeError):
    """Base for any reason a sealed tool MUST NOT fire on the eager path.

    Adapters catch this to route the call elsewhere — record an error,
    fall through to the framework's tool step, etc. The original cause (if
    any) is chained on `__cause__`.
    """


class NonIdempotentToolError(EagerDispatchDeniedError):
    """Raised when a non-idempotent tool is offered to the eager executor.

    Non-idempotent tools (payments, destructive CLI commands, outbound messages)
    must NOT fire before `message_stop` — the model may retract them mid-stream.
    Callers should catch this (or its base `EagerDispatchDeniedError`) and route to
    the classic dispatch path.
    """


class GateDeniedError(EagerDispatchDeniedError):
    """Raised when a tool's per-call `gate(call)` denies eager dispatch.

    Either `gate` returned False (denial) or `gate` raised (treated as denial,
    with the original exception chained on `__cause__`). Sibling — not
    subclass — of `NonIdempotentToolError`: gate denial is a per-call,
    args-aware decision; idempotency is a per-tool blanket policy.
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
        self._sem = asyncio.Semaphore(max_concurrent)
        self._in_flight: set[asyncio.Task[None]] = set()
        self._results: asyncio.Queue[tuple[ToolCall, Any | Exception] | None] = asyncio.Queue()
        self._closed = False

    async def record_error(self, call: ToolCall, exc: BaseException) -> None:
        """Record an externally-detected error against `call` without dispatching it.

        Used by adapters to surface seal-time parse errors (malformed JSON args,
        missing tool name) into the same `results()` channel that real tool
        results flow through. The pool itself never "ran" the tool.
        """
        if self._closed:
            raise RuntimeError("ExecutorPool is closed")
        await self._results.put((call, exc))

    async def dispatch(self, call: ToolCall) -> None:
        """Fire a sealed tool. Raises `EagerDispatchDeniedError` if the call cannot
        run on the eager path (non-idempotent tool, or per-call gate denied).

        Returns immediately — the tool runs on a background task. Results arrive
        via `results()`.

        On denial, the pool does NOT push to `_results`. The caller decides
        whether to record the error, fall through to a framework's tool step,
        or ignore the call entirely.

        Cancellation: if the calling task is cancelled while `await gate(call)`
        is suspended, the gate task receives `CancelledError` and propagates
        per asyncio semantics — same contract as tool execution. Gates with
        side effects (queue inserts, audit logs) must clean up in their own
        `try/finally`.
        """
        if self._closed:
            raise RuntimeError("ExecutorPool is closed")
        tool = self._tools.get(call.name)
        if tool is None:
            await self._results.put((call, KeyError(f"Unknown tool: {call.name}")))
            return
        if not tool.idempotent:
            raise NonIdempotentToolError(
                f"Tool {call.name!r} is not idempotent; eager dispatch forbidden."
            )
        gate = getattr(tool, "gate", None)
        if gate is not None:
            try:
                allowed = await gate(call)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise GateDeniedError(
                    f"Gate for {call.name!r} raised; eager dispatch denied."
                ) from exc
            if not allowed:
                raise GateDeniedError(
                    f"Gate for {call.name!r} returned False; eager dispatch denied."
                )
        self._safe_hook("on_dispatch_start", call)
        task = asyncio.create_task(self._run_one(call, tool), name=f"eager-{call.tool_call_id}")
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)

    async def _run_one(self, call: ToolCall, tool: Tool) -> None:
        async with self._sem:
            try:
                result = await tool(call.arguments)
                await self._results.put((call, result))
                self._safe_hook("on_dispatch_end", call, None)
            except asyncio.CancelledError:
                self._safe_hook("on_dispatch_end", call, None)
                raise
            except Exception as exc:
                await self._results.put((call, exc))
                self._safe_hook("on_dispatch_end", call, exc)

    async def results(self) -> AsyncIterator[tuple[ToolCall, Any | Exception]]:
        """Async iterator over completed tool results, in completion order.

        Yields `(ToolCall, result)` on success, `(ToolCall, Exception)` on failure.
        Iteration ends when `close()` is called and remaining tasks drain.
        """
        while True:
            item = await self._results.get()
            if item is None:
                return
            yield item

    async def cancel_all(self) -> None:
        """Cancel every in-flight task. Idempotent. Safe to call from any scope."""
        for task in list(self._in_flight):
            task.cancel()
        if self._in_flight:
            await asyncio.gather(*self._in_flight, return_exceptions=True)
        self._in_flight.clear()

    async def close(self) -> None:
        """Signal no more dispatches will arrive. `results()` drains then closes."""
        if self._closed:
            return
        self._closed = True
        if self._in_flight:
            await asyncio.gather(*self._in_flight, return_exceptions=True)
        await self._results.put(None)

    @property
    def in_flight(self) -> int:
        """Count of tool tasks currently running."""
        return len(self._in_flight)

    def _safe_hook(self, name: str, *args: Any) -> None:
        with contextlib.suppress(Exception):
            getattr(self._observability, name)(*args)


__all__ = [
    "EagerDispatchDeniedError",
    "ExecutorPool",
    "GateDeniedError",
    "NonIdempotentToolError",
]
