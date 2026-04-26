"""AnthropicEagerStream — wraps an Anthropic Messages stream and yields SealEvents.

Owns a `SealDetector` + `ExecutorPool` internally. The public surface is:

- `events()` — async iterator of `SealEvent`s as tools seal
- `results()` — async iterator of `(ToolCall, result | Exception)` as tools complete
- `cancel()` — tear everything down; propagates to the executor pool

See METHOD.md §4 for the full runtime contract.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from eager_tools import (
    NOOP_OBSERVABILITY,
    ExecutorPool,
    ObservabilityHook,
    SealDetector,
    SealEvent,
    Tool,
    ToolCall,
)

from .chunks import normalize_event


class AnthropicEagerStream:
    """Adapter: Anthropic Messages stream → eager tool dispatch.

    Usage:

        async with anthropic_client.messages.stream(...) as source:
            runner = AnthropicEagerStream(source, tools=my_tools)
            async for event in runner.events():
                ...
            async for call, result in runner.results():
                ...
    """

    def __init__(
        self,
        source: AsyncIterator[Any],
        tools: dict[str, Tool],
        *,
        observability: ObservabilityHook = NOOP_OBSERVABILITY,
        max_concurrent: int = 32,
        conversation_id: str | None = None,
    ) -> None:
        self._source = source
        self._observability = observability
        self._detector = SealDetector(conversation_id=conversation_id)
        self._pool = ExecutorPool(
            tools,
            observability=observability,
            max_concurrent=max_concurrent,
        )

    async def events(self) -> AsyncIterator[SealEvent]:
        """Async iterator of SealEvents. Consumes `source` once.

        On `message_stop`: seals the final in-flight tool, yields a
        `message_complete` event, then closes the executor pool so `results()`
        terminates cleanly. On cancellation: cancels in-flight tools and
        re-raises.
        """
        try:
            async for event in self._source:
                chunk = normalize_event(event)
                if chunk is not None:
                    seal = self._detector.observe(
                        tool_call_id=chunk.tool_call_id,
                        index=chunk.index,
                        name=chunk.name,
                        args_delta=chunk.args_delta,
                    )
                    if seal is not None and seal.tool_call is not None:
                        await self._handle_seal(seal)
                        self._fire_on_seal(seal)
                        yield seal

                if getattr(event, "type", None) == "message_stop":
                    final = self._detector.finalize()
                    if final is not None and final.tool_call is not None:
                        await self._handle_seal(final)
                        self._fire_on_seal(final)
                        yield final
                    msg_complete = SealEvent(kind="message_complete")
                    self._fire_on_seal(msg_complete)
                    yield msg_complete
                    await self._pool.close()
                    return

            # Source ended without an explicit message_stop event.
            final = self._detector.finalize()
            if final is not None and final.tool_call is not None:
                await self._handle_seal(final)
                self._fire_on_seal(final)
                yield final
            msg_complete = SealEvent(kind="message_complete")
            self._fire_on_seal(msg_complete)
            yield msg_complete
            await self._pool.close()
        except (asyncio.CancelledError, GeneratorExit):
            await self._pool.cancel_all()
            await self._pool.close()
            raise

    def _fire_on_seal(self, seal: SealEvent) -> None:
        """Notify the observability hook (sync — `ObservabilityHook.on_seal`
        is `def`, not `async def`). Suppresses observer errors so a broken
        hook can never abort the stream — same semantics as
        `ExecutorPool._safe_hook`.
        """
        with contextlib.suppress(Exception):
            self._observability.on_seal(seal)

    async def _handle_seal(self, seal: SealEvent) -> None:
        """Dispatch a sealed tool, or push the parse_error onto results without dispatching.

        When `seal.parse_error` is set, the tool block was malformed (bad JSON,
        missing name) — the call is unsafe to dispatch but the error needs to
        reach `results()` so the caller can surface it as an error tool message.
        """
        assert seal.tool_call is not None
        if seal.parse_error is not None:
            await self._pool.record_error(seal.tool_call, seal.parse_error)
            return
        await self._pool.dispatch(seal.tool_call)

    async def results(self) -> AsyncIterator[tuple[ToolCall, Any | Exception]]:
        """Async iterator of completed tool results, in completion order.

        Terminates when `events()` (or `cancel()`) closes the underlying pool.
        Iterate concurrently with `events()` via `asyncio.gather` / `TaskGroup`,
        or drain `events()` first then `results()` for offline replay.
        """
        async for item in self._pool.results():
            yield item

    async def cancel(self) -> None:
        """Tear down the stream + in-flight tool tasks. Idempotent."""
        await self._pool.cancel_all()
        await self._pool.close()


__all__ = ["AnthropicEagerStream"]
