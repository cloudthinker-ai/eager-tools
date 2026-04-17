"""SimpleNamespace fixture builders for Anthropic stream events.

Used by the replay tests to feed synthetic events without importing the
anthropic SDK. Mirrors the style in `eager-tools-core/tests/`.
"""

from __future__ import annotations

from types import SimpleNamespace


def tool_use_start(idx: int, tool_id: str, name: str) -> SimpleNamespace:
    """`content_block_start` event for a `tool_use` block."""
    return SimpleNamespace(
        type="content_block_start",
        index=idx,
        content_block=SimpleNamespace(type="tool_use", id=tool_id, name=name),
    )


def text_block_start(idx: int) -> SimpleNamespace:
    """`content_block_start` for a `text` block — adapter must ignore."""
    return SimpleNamespace(
        type="content_block_start",
        index=idx,
        content_block=SimpleNamespace(type="text"),
    )


def input_json_delta(idx: int, partial: str) -> SimpleNamespace:
    """`content_block_delta` carrying an `input_json_delta`."""
    return SimpleNamespace(
        type="content_block_delta",
        index=idx,
        delta=SimpleNamespace(type="input_json_delta", partial_json=partial),
    )


def text_delta(idx: int, text: str) -> SimpleNamespace:
    """`content_block_delta` carrying a `text_delta` — adapter must ignore."""
    return SimpleNamespace(
        type="content_block_delta",
        index=idx,
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def content_block_stop(idx: int) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_stop", index=idx)


def message_start() -> SimpleNamespace:
    return SimpleNamespace(type="message_start")


def message_stop() -> SimpleNamespace:
    return SimpleNamespace(type="message_stop")


def ping() -> SimpleNamespace:
    return SimpleNamespace(type="ping")
