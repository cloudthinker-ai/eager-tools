"""OpenAIEagerStream replay tests.

All skipped — bodies land during Move 3. Test names + docstrings document
target behavior. Tests feed synthetic chunks (SimpleNamespace, no SDK import)
to keep CI fast, deterministic, and offline.

Source of truth: METHOD.md §3 + the core SealDetector contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from eager_tools import Tool
from eager_tools_openai import OpenAIEagerStream
from eager_tools_openai.chunks import normalize_chunk

pytestmark = pytest.mark.skip(reason="Implementation deferred to Move 3 (port phase).")


def _chunk(tool_calls: list[SimpleNamespace], finish_reason: str | None = None) -> SimpleNamespace:
    """Build a minimal OpenAI-shape chunk with the given tool_call deltas."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
    )


def test_tool_calls_delta_first_emits_id_name() -> None:
    """First delta for a tool_call slot carries `id` + `function.name`."""
    chunk = _chunk([
        SimpleNamespace(
            index=0,
            id="call_A",
            function=SimpleNamespace(name="read_file", arguments=""),
        )
    ])
    out = normalize_chunk(chunk)
    assert len(out) == 1
    assert out[0].tool_call_id == "call_A"
    assert out[0].name == "read_file"
    assert out[0].index == 0
    assert out[0].args_delta == ""


def test_tool_calls_delta_args_routes_by_index() -> None:
    """Subsequent args delta lacks `id` — normalizer routes by `index`."""
    chunk = _chunk([
        SimpleNamespace(
            index=0,
            id=None,
            function=SimpleNamespace(name=None, arguments='{"path":'),
        )
    ])
    out = normalize_chunk(chunk)
    assert len(out) == 1
    assert out[0].tool_call_id is None
    assert out[0].index == 0
    assert out[0].args_delta == '{"path":'


async def test_end_to_end_two_tool_sequence() -> None:
    """Replay two sequential tool_call deltas; expect 2 SealEvents with kind='tool_sealed'."""
    raw_chunks = [
        _chunk([
            SimpleNamespace(
                index=0,
                id="call_A",
                function=SimpleNamespace(name="read_file", arguments=""),
            )
        ]),
        _chunk([
            SimpleNamespace(
                index=0,
                id=None,
                function=SimpleNamespace(name=None, arguments='{"path":"/a"}'),
            )
        ]),
        _chunk([
            SimpleNamespace(
                index=1,
                id="call_B",
                function=SimpleNamespace(name="http_get", arguments=""),
            )
        ]),
        _chunk([
            SimpleNamespace(
                index=1,
                id=None,
                function=SimpleNamespace(name=None, arguments='{"url":"/b"}'),
            )
        ]),
        _chunk([], finish_reason="tool_calls"),
    ]

    async def source():
        for c in raw_chunks:
            yield c

    tools: dict[str, Tool] = {}
    stream = OpenAIEagerStream(source(), tools=tools)
    seals = [ev async for ev in stream.events()]
    assert len(seals) == 2
    assert all(s.kind == "tool_sealed" for s in seals)
