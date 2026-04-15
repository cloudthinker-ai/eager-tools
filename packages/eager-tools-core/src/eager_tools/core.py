"""Seal detector — the state machine that turns streaming chunks into SealEvents.

Adapters feed raw chunks via `observe(...)` as they arrive from the provider
stream. The detector emits a SealEvent the moment a new `tool_call_id` appears
(sealing the previous tool) or `finalize()` is called on message_stop.

This module is provider-agnostic. All provider-specific chunk shapes are
normalized by adapters before reaching here.

See METHOD.md §3 for the underlying mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import SealEvent, ToolCall


@dataclass(slots=True)
class _ToolBuffer:
    """Internal — accumulates partial arg JSON for one tool_call_id."""

    tool_call_id: str
    name: str | None = None
    args_chunks: list[str] = field(default_factory=list)

    def materialize(self, conversation_id: str | None) -> ToolCall:
        """Parse accumulated arg chunks into a final ToolCall."""
        _ = conversation_id
        raise NotImplementedError("Implementation deferred to Move 3 (port phase).")


class SealDetector:
    """State machine: streaming chunks → SealEvents.

    Usage (from an adapter):

        detector = SealDetector(conversation_id="conv-123")

        async for chunk in provider_stream:
            event = detector.observe(
                tool_call_id=chunk.tool_call_id,
                index=chunk.index,
                name=chunk.name,
                args_delta=chunk.delta,
            )
            if event is not None:
                yield event  # tool_sealed — dispatch now

        final = detector.finalize()
        if final is not None:
            yield final  # seals the last in-flight tool
    """

    def __init__(self, conversation_id: str | None = None) -> None:
        self._conversation_id = conversation_id
        self._last_tool_call_id: str | None = None
        self._buffers: dict[str, _ToolBuffer] = {}
        self._index_to_id: dict[int, str] = {}

    def observe(
        self,
        *,
        tool_call_id: str | None,
        index: int | None,
        name: str | None,
        args_delta: str,
    ) -> SealEvent | None:
        """Feed one chunk. Returns a SealEvent iff this chunk sealed a previous tool.

        `tool_call_id` is present on a tool's FIRST chunk only. Subsequent chunks
        carry only `index` — the detector routes them back to the correct buffer
        via the index→id map built on the first chunk.
        """
        _ = (tool_call_id, index, name, args_delta)
        raise NotImplementedError("Implementation deferred to Move 3 (port phase).")

    def finalize(self) -> SealEvent | None:
        """Called on message_stop. Seals the final in-flight tool, if any.

        Returns None if there was no in-flight tool (tool-less message).
        """
        raise NotImplementedError("Implementation deferred to Move 3 (port phase).")

    def reset(self) -> None:
        """Reset detector state between conversations. Cheaper than re-instantiating."""
        self._last_tool_call_id = None
        self._buffers.clear()
        self._index_to_id.clear()

    @property
    def in_flight_tool_call_id(self) -> str | None:
        """The tool_call_id currently accumulating chunks, if any."""
        return self._last_tool_call_id


__all__ = ["SealDetector"]
