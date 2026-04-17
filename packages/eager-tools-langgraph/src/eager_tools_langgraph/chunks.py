"""LangChain chunk normalizer — `AIMessageChunk` → core chunk shape.

Pure functions only. No I/O. `normalize_chunk` maps one `AIMessageChunk` into
zero or more `NormalizedChunk`s — one per `tool_call_chunk` in the message.

Why a list, not a single value: a single `AIMessageChunk` can carry multiple
`tool_call_chunks` (parallel tool calls in one streaming step). We surface them
in the order the provider emitted them.

Why correlate by `index`, not `id`: chunks beyond the first carry `id=None` for
some providers (Anthropic via langchain-anthropic, OpenAI). The `index` field
is the only stable correlator across the whole stream.

See `eager_tools.SealDetector.observe(...)` for the four-field contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NormalizedChunk:
    """The 4 fields `SealDetector.observe(...)` consumes."""

    tool_call_id: str | None
    index: int | None
    name: str | None
    args_delta: str

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "index": self.index,
            "name": self.name,
            "args_delta": self.args_delta,
        }


def normalize_chunk(chunk: Any) -> list[NormalizedChunk]:
    """Map one `AIMessageChunk` → list of `NormalizedChunk`.

    Returns `[]` for chunks with no `tool_call_chunks` (pure text deltas, usage
    metadata, etc.). The detector consumes each item in order.
    """
    raw_chunks = getattr(chunk, "tool_call_chunks", None) or []
    out: list[NormalizedChunk] = []
    for tc in raw_chunks:
        out.append(
            NormalizedChunk(
                tool_call_id=tc.get("id"),
                index=tc.get("index"),
                name=tc.get("name"),
                args_delta=tc.get("args") or "",
            )
        )
    return out


__all__ = ["NormalizedChunk", "normalize_chunk"]
