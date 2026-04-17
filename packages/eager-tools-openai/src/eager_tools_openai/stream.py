"""OpenAIEagerStream — wraps an OpenAI chat.completions stream and yields SealEvents.

Owns a `SealDetector` + `ExecutorPool` internally. The public surface mirrors
`AnthropicEagerStream` exactly so runtimes can swap providers without changing
call sites.

See METHOD.md §4 for the full runtime contract.
"""

from __future__ import annotations

import asyncio
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

from .chunks import normalize_chunk

_FINISH_REASONS_END_STREAM = frozenset({"tool_calls", "stop", "length", "content_filter"})


class OpenAIEagerStream:
    """Adapter: OpenAI chat.completions streaming → eager tool dispatch.

    Usage:

        source = await client.chat.completions.create(..., stream=True)
        runner = OpenAIEagerStream(source, tools=my_tools)

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
        self._detector = SealDetector(conversation_id=conversation_id)
        self._pool = ExecutorPool(
            tools,
            observability=observability,
            max_concurrent=max_concurrent,
        )

    async def events(self) -> AsyncIterator[SealEvent]:
        """Async iterator of SealEvents. Consumes `source` once.

        On `finish_reason in {tool_calls, stop, length, content_filter}`: seals
        the final in-flight tool, yields a `message_complete` event, then closes
        the executor pool so `results()` terminates cleanly. On cancellation:
        cancels in-flight tools and re-raises.
        """
        try:
            async for chunk in self._source:
                for nc in normalize_chunk(chunk):
                    seal = self._detector.observe(
                        tool_call_id=nc.tool_call_id,
                        index=nc.index,
                        name=nc.name,
                        args_delta=nc.args_delta,
                    )
                    if seal is not None and seal.tool_call is not None:
                        await self._pool.dispatch(seal.tool_call)
                        yield seal

                if _stream_ended(chunk):
                    final = self._detector.finalize()
                    if final is not None and final.tool_call is not None:
                        await self._pool.dispatch(final.tool_call)
                        yield final
                    yield SealEvent(kind="message_complete")
                    await self._pool.close()
                    return

            # Source ended without a finish_reason — finalize anyway.
            final = self._detector.finalize()
            if final is not None and final.tool_call is not None:
                await self._pool.dispatch(final.tool_call)
                yield final
            yield SealEvent(kind="message_complete")
            await self._pool.close()
        except (asyncio.CancelledError, GeneratorExit):
            await self._pool.cancel_all()
            await self._pool.close()
            raise

    async def results(self) -> AsyncIterator[tuple[ToolCall, Any | Exception]]:
        """Async iterator of completed tool results, in completion order.

        Terminates when `events()` (or `cancel()`) closes the underlying pool.
        """
        async for item in self._pool.results():
            yield item

    async def cancel(self) -> None:
        """Tear down the stream + in-flight tool tasks. Idempotent."""
        await self._pool.cancel_all()
        await self._pool.close()


def _stream_ended(chunk: Any) -> bool:
    """True if the chunk's first choice carries a terminal `finish_reason`."""
    choices: list[Any] = getattr(chunk, "choices", None) or []
    if not choices:
        return False
    return getattr(choices[0], "finish_reason", None) in _FINISH_REASONS_END_STREAM


__all__ = ["OpenAIEagerStream"]
