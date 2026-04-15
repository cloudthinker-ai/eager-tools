"""AnthropicEagerStream replay tests.

All skipped — bodies land during Move 3. Test names + docstrings document
target behavior. Tests feed synthetic events (SimpleNamespace, no SDK import)
to keep CI fast, deterministic, and offline.

Source of truth: METHOD.md §3 + the core SealDetector contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from eager_tools import Tool
from eager_tools_anthropic import AnthropicEagerStream
from eager_tools_anthropic.chunks import normalize_event

pytestmark = pytest.mark.skip(reason="Implementation deferred to Move 3 (port phase).")


def test_tool_use_block_start_emits_first_chunk() -> None:
    """`content_block_start{type=tool_use, id, name}` → chunk with id + name populated."""
    event = SimpleNamespace(
        type="content_block_start",
        index=0,
        content_block=SimpleNamespace(type="tool_use", id="toolu_A", name="read_file"),
    )
    chunk = normalize_event(event)
    assert chunk is not None
    assert chunk.tool_call_id == "toolu_A"
    assert chunk.name == "read_file"
    assert chunk.index == 0
    assert chunk.args_delta == ""


def test_input_json_delta_emits_args_chunk() -> None:
    """`content_block_delta{type=input_json_delta}` → chunk with args_delta, no id."""
    event = SimpleNamespace(
        type="content_block_delta",
        index=0,
        delta=SimpleNamespace(type="input_json_delta", partial_json='{"path":'),
    )
    chunk = normalize_event(event)
    assert chunk is not None
    assert chunk.tool_call_id is None
    assert chunk.index == 0
    assert chunk.args_delta == '{"path":'


async def test_end_to_end_two_tool_sequence() -> None:
    """Replay two sequential tool blocks; expect 2 SealEvents with kind='tool_sealed'."""
    raw_events = [
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="tool_use", id="A", name="read_file"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"path":"/a"}'),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(
            type="content_block_start",
            index=1,
            content_block=SimpleNamespace(type="tool_use", id="B", name="http_get"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"url":"/b"}'),
        ),
        SimpleNamespace(type="content_block_stop", index=1),
        SimpleNamespace(type="message_stop"),
    ]

    async def source():
        for e in raw_events:
            yield e

    tools: dict[str, Tool] = {}
    stream = AnthropicEagerStream(source(), tools=tools)
    seals = [ev async for ev in stream.events()]
    assert len(seals) == 2
    assert all(s.kind == "tool_sealed" for s in seals)
