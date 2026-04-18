"""Async executor pool for sealed tool calls.

The pool's lifetime is owned by the stream reader. When the stream dies
(user interrupt, model stop sequence, network error), `cancel_all()` releases
every in-flight tool task cleanly.

Non-idempotent tools are rejected by `dispatch()` — callers must route them to
the classic (non-eager) path that fires after message_stop.

See METHOD.md §4 for the full runtime contract.
"""

from __future__ import annotations

import asyncio
import contextlib
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
        """Fire a sealed tool. Raises NonIdempotentToolError for unsafe tools.

        Returns immediately — the tool runs on a background task. Results arrive
        via `results()`.
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


__all__ = ["ExecutorPool", "NonIdempotentToolError"]
