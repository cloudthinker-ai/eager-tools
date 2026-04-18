"""Seal detector — the state machine that turns streaming chunks into SealEvents.

Adapters feed raw chunks via `observe(...)` as they arrive from the provider
stream. The detector emits a SealEvent the moment a new `tool_call_id` appears
(sealing the previous tool) or `finalize()` is called on message_stop.

This module is provider-agnostic. All provider-specific chunk shapes are
normalized by adapters before reaching here.

See METHOD.md §3 for the underlying mechanism.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .types import SealEvent, ToolCall


@dataclass(slots=True)
class _ToolBuffer:
    """Internal — accumulates partial arg JSON for one tool_call_id."""

    tool_call_id: str
    name: str | None = None
    args_chunks: list[str] = field(default_factory=list)

    def materialize(self, conversation_id: str | None) -> ToolCall:
        """Parse accumulated arg chunks into a final ToolCall."""
        if self.name is None:
            raise ValueError(f"Tool call {self.tool_call_id} has no name")
        raw = "".join(self.args_chunks).strip()
        arguments: dict[str, Any] = json.loads(raw) if raw else {}
        return ToolCall(
            tool_call_id=self.tool_call_id,
            name=self.name,
            arguments=arguments,
            conversation_id=conversation_id,
        )

    def materialize_partial(self, conversation_id: str | None) -> ToolCall:
        """Best-effort ToolCall used when materialize() fails — id known, args empty."""
        return ToolCall(
            tool_call_id=self.tool_call_id,
            name=self.name or "",
            arguments={},
            conversation_id=conversation_id,
        )


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
        # perf_counter() timestamp when first chunk arrived for each buffer
        self._buffer_start_ts: dict[str, float] = {}

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
        if tool_call_id is not None:
            if tool_call_id == self._last_tool_call_id:
                buf = self._buffers[tool_call_id]
                if name is not None:
                    buf.name = name
                if args_delta:
                    buf.args_chunks.append(args_delta)
                if index is not None:
                    self._index_to_id[index] = tool_call_id
                return None

            sealed_event: SealEvent | None = None
            if self._last_tool_call_id is not None:
                sealed_event = self._seal(self._last_tool_call_id)

            self._buffers[tool_call_id] = _ToolBuffer(tool_call_id, name)
            self._buffer_start_ts[tool_call_id] = time.perf_counter()
            if index is not None:
                self._index_to_id[index] = tool_call_id
            if args_delta:
                self._buffers[tool_call_id].args_chunks.append(args_delta)
            self._last_tool_call_id = tool_call_id
            return sealed_event

        if index is not None and index in self._index_to_id:
            target_id = self._index_to_id[index]
        elif self._last_tool_call_id is not None:
            target_id = self._last_tool_call_id
        else:
            return None

        buf = self._buffers.get(target_id)
        if buf is None:
            return None
        if name is not None:
            buf.name = name
        if args_delta:
            buf.args_chunks.append(args_delta)
        return None

    def _seal(self, tool_call_id: str) -> SealEvent:
        buf = self._buffers.pop(tool_call_id)
        start_ts = self._buffer_start_ts.pop(tool_call_id, None)
        latency_ms = (time.perf_counter() - start_ts) * 1000.0 if start_ts is not None else 0.0
        try:
            tool_call = buf.materialize(self._conversation_id)
        except (ValueError, json.JSONDecodeError) as exc:
            # Materialization failed (malformed JSON / missing name). Emit the
            # seal anyway with parse_error set so the adapter can surface a
            # tool error to the model rather than crashing the whole stream.
            return SealEvent(
                kind="tool_sealed",
                tool_call=buf.materialize_partial(self._conversation_id),
                seal_latency_ms=latency_ms,
                parse_error=exc,
            )
        return SealEvent(
            kind="tool_sealed",
            tool_call=tool_call,
            seal_latency_ms=latency_ms,
        )

    def finalize(self) -> SealEvent | None:
        """Called on message_stop. Seals the final in-flight tool, if any.

        Returns None if there was no in-flight tool (tool-less message).
        """
        if self._last_tool_call_id is None:
            return None
        event = self._seal(self._last_tool_call_id)
        self._last_tool_call_id = None
        return event

    def reset(self) -> None:
        """Reset detector state between conversations. Cheaper than re-instantiating."""
        self._last_tool_call_id = None
        self._buffers.clear()
        self._index_to_id.clear()
        self._buffer_start_ts.clear()

    @property
    def in_flight_tool_call_id(self) -> str | None:
        """The tool_call_id currently accumulating chunks, if any."""
        return self._last_tool_call_id


__all__ = ["SealDetector"]
