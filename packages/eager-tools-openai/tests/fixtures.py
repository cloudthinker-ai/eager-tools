"""SimpleNamespace fixture builders for OpenAI chat.completions chunks.

Used by the replay tests to feed synthetic chunks without importing the
openai SDK. Mirrors the style in `eager-tools-core/tests/`.
"""

from __future__ import annotations

from types import SimpleNamespace


def _delta(tool_calls: list[SimpleNamespace] | None, content: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(tool_calls=tool_calls, content=content)


def _choice(delta: SimpleNamespace, finish_reason: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)


def chunk_with_tool_calls(
    tool_calls: list[SimpleNamespace], finish_reason: str | None = None
) -> SimpleNamespace:
    """A chat-completion chunk carrying one or more `tool_calls` deltas."""
    return SimpleNamespace(choices=[_choice(_delta(tool_calls), finish_reason)])


def tool_call_first(index: int, tc_id: str, name: str, args: str = "") -> SimpleNamespace:
    """A `ChoiceDeltaToolCall` carrying id + function.name (FIRST delta for slot)."""
    return SimpleNamespace(
        index=index,
        id=tc_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=args),
    )


def tool_call_args(index: int, args: str) -> SimpleNamespace:
    """A continuation delta carrying only args (no id, no name)."""
    return SimpleNamespace(
        index=index,
        id=None,
        function=SimpleNamespace(name=None, arguments=args),
    )


def text_chunk(content: str) -> SimpleNamespace:
    """A text-only chunk (no tool_calls) — adapter must ignore."""
    return SimpleNamespace(choices=[_choice(_delta(None, content=content))])


def finish(reason: str = "tool_calls") -> SimpleNamespace:
    """Final chunk with `finish_reason` set and no tool_calls deltas."""
    return SimpleNamespace(choices=[_choice(_delta(None), finish_reason=reason)])
