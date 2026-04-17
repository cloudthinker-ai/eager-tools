"""Anthropic chunk normalizer — SDK events → core chunk shape.

Pure functions only. No I/O. `normalize_event` maps one Anthropic stream event
into the 4-tuple `SealDetector.observe(...)` consumes, or returns None for
events that don't carry tool-call info (text deltas, message metadata, ping).

Target Anthropic event shapes (Messages streaming API):

- `content_block_start` with `content_block.type == "tool_use"`
    → first chunk for a tool, carrying `id` + `name`
- `content_block_delta` with `delta.type == "input_json_delta"`
    → subsequent args chunk, carrying `partial_json` (no id)
- `content_block_stop` / `message_stop` / text deltas
    → return None (not tool-related at the chunk layer)

See METHOD.md §3 for the core chunk contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NormalizedChunk:
    """The 4 fields SealDetector.observe(...) consumes."""

    tool_call_id: str | None
    index: int | None
    name: str | None
    args_delta: str


def normalize_event(event: Any) -> NormalizedChunk | None:
    """Map one Anthropic stream event → NormalizedChunk, or None if not tool-related.

    Returns None for text content deltas, message_start, message_stop, ping, and
    content_block_stop (the stop event itself is redundant — the SealDetector
    detects seals via the next tool's first chunk).
    """
    event_type = getattr(event, "type", None)

    if event_type == "content_block_start":
        block = getattr(event, "content_block", None)
        if getattr(block, "type", None) != "tool_use":
            return None
        return NormalizedChunk(
            tool_call_id=getattr(block, "id", None),
            index=getattr(event, "index", None),
            name=getattr(block, "name", None),
            args_delta="",
        )

    if event_type == "content_block_delta":
        delta = getattr(event, "delta", None)
        if getattr(delta, "type", None) != "input_json_delta":
            return None
        return NormalizedChunk(
            tool_call_id=None,
            index=getattr(event, "index", None),
            name=None,
            args_delta=getattr(delta, "partial_json", "") or "",
        )

    return None


__all__ = ["NormalizedChunk", "normalize_event"]
