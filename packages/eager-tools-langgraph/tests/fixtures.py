"""AIMessageChunk fixture builders for replay tests.

Mirrors `packages/eager-tools-anthropic/tests/fixtures.py` — small helpers that
build the exact streaming shape `langchain-anthropic` / `langchain-openai`
emit, without importing either provider SDK.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessageChunk


def tool_chunk(
    *,
    index: int,
    name: str | None = None,
    args: str = "",
    tool_id: str | None = None,
) -> AIMessageChunk:
    """One streamed `AIMessageChunk` carrying a single `tool_call_chunk`.

    First chunk for a tool call carries `tool_id` + `name`; subsequent chunks
    set both to `None` and only stream `args` deltas. The `index` is the only
    stable correlator.
    """
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": name,
                "args": args,
                "id": tool_id,
                "index": index,
            }
        ],
    )


def text_chunk(text: str) -> AIMessageChunk:
    """Plain text token — adapter must not produce a NormalizedChunk for this."""
    return AIMessageChunk(content=text)


def parallel_tool_chunk(
    *specs: dict[str, Any],
) -> AIMessageChunk:
    """A single `AIMessageChunk` carrying multiple `tool_call_chunks` at once.

    Each spec is `{"index": int, "name": str|None, "args": str, "id": str|None}`.
    """
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": s.get("name"),
                "args": s.get("args", ""),
                "id": s.get("id"),
                "index": s["index"],
            }
            for s in specs
        ],
    )


def script(*chunks: AIMessageChunk) -> ScriptedStream:
    """Wrap a chunk sequence in an async iterator + a fake model object.

    Returned object exposes `astream(messages, **kwargs)` — pluggable into
    `ModelRequest.model` so the middleware can drive it like a real model.
    """
    return ScriptedStream(list(chunks))


class ScriptedStream:
    """Stand-in for a `BaseChatModel` that yields a fixed chunk sequence."""

    def __init__(self, chunks: list[AIMessageChunk]) -> None:
        self._chunks = chunks
        self.calls: int = 0

    def astream(self, messages: Any, **_kwargs: Any) -> AsyncIterator[AIMessageChunk]:
        del messages
        self.calls += 1
        return _aiter(self._chunks)


async def _aiter(items: list[AIMessageChunk]) -> AsyncIterator[AIMessageChunk]:
    for it in items:
        yield it
