"""OpenAIEagerStream — wraps an OpenAI chat.completions stream and yields SealEvents.

Owns a `SealDetector` + `ExecutorPool` internally. The public surface mirrors
`AnthropicEagerStream` exactly so runtimes can swap providers without changing
call sites.

See METHOD.md §4 for the full runtime contract.
"""

from __future__ import annotations

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
        """Async iterator of SealEvents. Consumes `source` once."""
        if False:  # pragma: no cover — keeps the signature an async generator
            yield  # type: ignore[unreachable]
        raise NotImplementedError("Implementation deferred to Move 3 (port phase).")

    async def results(self) -> AsyncIterator[tuple[ToolCall, Any | Exception]]:
        """Async iterator of completed tool results, in completion order."""
        if False:  # pragma: no cover — keeps the signature an async generator
            yield  # type: ignore[unreachable]
        raise NotImplementedError("Implementation deferred to Move 3 (port phase).")

    async def cancel(self) -> None:
        """Tear down the stream + in-flight tool tasks. Idempotent."""
        await self._pool.cancel_all()


__all__ = ["OpenAIEagerStream"]
