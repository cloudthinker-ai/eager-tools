"""Adversarial edge-case tests (ROADMAP §2.4, METHOD.md §6).

Covers: mid-chunk JSON, empty arg blocks, malformed JSON, heavily interleaved tools.
"""

from __future__ import annotations

import json

from eager_tools import SealDetector


def test_mid_chunk_json_accumulates_across_many_deltas() -> None:
    """10+ tiny args_delta chunks composing one JSON object."""
    detector = SealDetector()
    fragments = [
        '{"',
        "pa",
        "th",
        '":',
        ' "/',
        "etc",
        "/ho",
        "sts",
        '",',
        ' "m',
        "ode",
        '": ',
        '"r',
        '"}',
    ]
    detector.observe(tool_call_id="A", index=0, name="read_file", args_delta=fragments[0])
    for frag in fragments[1:]:
        detector.observe(tool_call_id=None, index=0, name=None, args_delta=frag)
    event = detector.finalize()
    assert event is not None
    assert event.tool_call is not None
    assert event.tool_call.arguments == {"path": "/etc/hosts", "mode": "r"}


def test_empty_arg_block_still_seals() -> None:
    """Tool call with empty args_delta throughout → arguments == {}."""
    detector = SealDetector()
    detector.observe(tool_call_id="A", index=0, name="get_time", args_delta="")
    event = detector.finalize()
    assert event is not None
    assert event.tool_call is not None
    assert event.tool_call.arguments == {}


def test_malformed_json_surfaces_as_parse_error_event() -> None:
    """Broken JSON args produce a SealEvent with parse_error set, not a crash.

    The adapter should convert this into an error tool message; the stream
    keeps going so the model can recover on the next turn.
    """
    detector = SealDetector()
    detector.observe(tool_call_id="A", index=0, name="bad_tool", args_delta='{"broken')
    event = detector.finalize()
    assert event is not None
    assert event.kind == "tool_sealed"
    assert isinstance(event.parse_error, json.JSONDecodeError)
    assert event.tool_call is not None
    assert event.tool_call.tool_call_id == "A"
    assert event.tool_call.name == "bad_tool"
    assert event.tool_call.arguments == {}


def test_malformed_json_mid_stream_does_not_block_subsequent_tools() -> None:
    """Tool A has bad JSON, tool B is well-formed: A seals with parse_error,
    B seals cleanly. The bad call doesn't poison the rest of the stream."""
    detector = SealDetector()
    detector.observe(tool_call_id="A", index=0, name="bad", args_delta='{"x":')
    seal_a = detector.observe(tool_call_id="B", index=1, name="ok", args_delta='{"y":1}')
    assert seal_a is not None
    assert seal_a.parse_error is not None
    assert seal_a.tool_call is not None and seal_a.tool_call.tool_call_id == "A"

    seal_b = detector.finalize()
    assert seal_b is not None
    assert seal_b.parse_error is None
    assert seal_b.tool_call is not None
    assert seal_b.tool_call.tool_call_id == "B"
    assert seal_b.tool_call.arguments == {"y": 1}


def test_missing_name_surfaces_as_parse_error_event() -> None:
    """A tool block that never received a name still seals — with parse_error."""
    detector = SealDetector()
    # Provider sent index/args but never a `name` field.
    detector.observe(tool_call_id="A", index=0, name=None, args_delta='{"k":1}')
    event = detector.finalize()
    assert event is not None
    assert isinstance(event.parse_error, ValueError)
    assert event.tool_call is not None
    assert event.tool_call.tool_call_id == "A"
    assert event.tool_call.name == ""


def test_many_interleaved_tools() -> None:
    """5 tools A-E, each with split args, seal in correct order."""
    detector = SealDetector()
    seals: list[str] = []

    detector.observe(tool_call_id="A", index=0, name="t0", args_delta='{"k":')
    detector.observe(tool_call_id=None, index=0, name=None, args_delta='"a"}')

    ev = detector.observe(tool_call_id="B", index=1, name="t1", args_delta='{"k":')
    assert ev is not None
    seals.append(ev.tool_call.tool_call_id)  # type: ignore[union-attr]
    detector.observe(tool_call_id=None, index=1, name=None, args_delta='"b"}')

    ev = detector.observe(tool_call_id="C", index=2, name="t2", args_delta='{"k":')
    assert ev is not None
    seals.append(ev.tool_call.tool_call_id)  # type: ignore[union-attr]
    detector.observe(tool_call_id=None, index=2, name=None, args_delta='"c"}')

    ev = detector.observe(tool_call_id="D", index=3, name="t3", args_delta='{"k":')
    assert ev is not None
    seals.append(ev.tool_call.tool_call_id)  # type: ignore[union-attr]
    detector.observe(tool_call_id=None, index=3, name=None, args_delta='"d"}')

    ev = detector.observe(tool_call_id="E", index=4, name="t4", args_delta='{"k":')
    assert ev is not None
    seals.append(ev.tool_call.tool_call_id)  # type: ignore[union-attr]
    detector.observe(tool_call_id=None, index=4, name=None, args_delta='"e"}')

    final = detector.finalize()
    assert final is not None
    seals.append(final.tool_call.tool_call_id)  # type: ignore[union-attr]

    assert seals == ["A", "B", "C", "D", "E"]
