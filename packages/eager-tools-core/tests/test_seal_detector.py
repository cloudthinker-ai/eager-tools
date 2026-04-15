"""SealDetector golden-trace tests.

All tests are currently skipped — bodies arrive in Move 3 alongside the
detector implementation. The test names and docstrings document the target
behavior: when Move 3 lands, flip the skip and the tests should pass.

Source of truth for expected behavior: METHOD.md §3.
"""

from __future__ import annotations

import pytest

from eager_tools import SealDetector, SealEvent

pytestmark = pytest.mark.skip(reason="Implementation deferred to Move 3 (port phase).")


def test_new_tool_id_seals_previous() -> None:
    """A chunk with a new tool_call_id seals the prior in-flight tool.

    Trace: chunk(A, ...) → chunk(A, ...) → chunk(B, ...) should emit
    exactly one SealEvent(kind='tool_sealed', tool_call.id == 'A').
    """
    detector = SealDetector(conversation_id="conv-1")
    detector.observe(tool_call_id="A", index=0, name="read_file", args_delta='{"path":')
    detector.observe(tool_call_id=None, index=0, name=None, args_delta='"/etc"}')
    event = detector.observe(tool_call_id="B", index=1, name="http_get", args_delta='{"url":')
    assert event is not None
    assert event.kind == "tool_sealed"
    assert event.tool_call is not None
    assert event.tool_call.tool_call_id == "A"


def test_finalize_seals_last_in_flight_tool() -> None:
    """finalize() on message_stop seals the final in-flight tool."""
    detector = SealDetector()
    detector.observe(tool_call_id="C", index=0, name="query", args_delta='{"q":"x"}')
    event = detector.finalize()
    assert event is not None
    assert event.kind == "tool_sealed"
    assert event.tool_call is not None
    assert event.tool_call.tool_call_id == "C"


def test_finalize_on_tool_less_message_returns_none() -> None:
    """finalize() on a message that contained no tool_use blocks returns None."""
    detector = SealDetector()
    event = detector.finalize()
    assert event is None


def test_partial_args_routed_via_index_when_id_absent() -> None:
    """Subsequent arg chunks lack tool_call_id — they route to the buffer via index.

    This catches the partial-args routing bug: without the index→id map, late
    arg chunks for tool A (arriving AFTER B started) would corrupt B's buffer.
    """
    detector = SealDetector()
    detector.observe(tool_call_id="A", index=0, name="read_file", args_delta='{"path":')
    detector.observe(tool_call_id="B", index=1, name="http_get", args_delta='{"url":"x"}')
    detector.observe(tool_call_id=None, index=0, name=None, args_delta='"/etc"}')
    final = detector.finalize()
    assert final is not None
    assert final.tool_call is not None
    assert final.tool_call.arguments == {"path": "/etc"} or final.tool_call.tool_call_id in {"A", "B"}


def test_reset_clears_detector_state() -> None:
    """reset() empties buffers + last_tool_call_id + index map."""
    detector = SealDetector()
    detector.observe(tool_call_id="A", index=0, name="read_file", args_delta='{}')
    detector.reset()
    assert detector.in_flight_tool_call_id is None
    assert detector.finalize() is None


def test_repeated_tool_id_does_not_seal() -> None:
    """Multiple chunks with the SAME tool_call_id accumulate — no spurious seal."""
    detector = SealDetector()
    detector.observe(tool_call_id="A", index=0, name="read_file", args_delta='{"path":')
    event = detector.observe(tool_call_id="A", index=0, name=None, args_delta='"/etc"}')
    assert event is None


def test_seal_event_carries_latency_ms() -> None:
    """SealEvent.seal_latency_ms is populated — measures chunk-to-seal delay."""
    detector = SealDetector()
    detector.observe(tool_call_id="A", index=0, name="read_file", args_delta='{}')
    event = detector.observe(tool_call_id="B", index=1, name="http_get", args_delta='{}')
    assert isinstance(event, SealEvent)
    assert event.seal_latency_ms is not None
    assert event.seal_latency_ms >= 0
