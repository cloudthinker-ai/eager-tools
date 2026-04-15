"""OpenAI chunk normalizer — chat.completions chunks → core chunk shape.

Pure functions only. No I/O. `normalize_chunk` maps one OpenAI
`ChatCompletionChunk` into zero-or-more `NormalizedChunk`s (one per tool_call
delta carried in the chunk), or returns an empty list if the chunk carries no
tool-call info.

Target OpenAI chunk shape (chat.completions streaming API):

- `chunk.choices[0].delta.tool_calls[i]` — list of `ChoiceDeltaToolCall`
    - `.index`                           : int, the tool's slot in the array
    - `.id`                              : str, present on the FIRST delta for a slot
    - `.function.name`                   : str, present on the first delta
    - `.function.arguments`              : str, cumulative args JSON fragment

The final chunk carries `choices[0].finish_reason == "tool_calls"` — the
SealDetector relies on the next tool's id (or stream end) for sealing, so
`finish_reason` alone is not a per-block signal.

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


def normalize_chunk(chunk: Any) -> list[NormalizedChunk]:
    """Map one OpenAI chat completion chunk → 0..N NormalizedChunks.

    A single chunk can theoretically carry deltas for multiple tool_calls
    (different indices), so this returns a list. Empty list for chunks with
    no `delta.tool_calls` (text deltas, role deltas, finish-only chunks).
    """
    _ = chunk
    raise NotImplementedError("Implementation deferred to Move 3 (port phase).")


__all__ = ["NormalizedChunk", "normalize_chunk"]
