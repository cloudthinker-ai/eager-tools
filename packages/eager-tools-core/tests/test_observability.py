# pyright: reportMissingImports=false, reportMissingTypeStubs=false
"""Tests for `OTelObservability`.

Skipped wholesale when `opentelemetry.sdk` is not installed — the runtime
[otel] extra ships only `opentelemetry-api`; the SDK lives in dev deps for
`InMemorySpanExporter`.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from eager_tools import OTelObservability, SealEvent, ToolCall


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    """Fresh in-memory exporter per test, wired to a fresh TracerProvider.

    Note: OTel only honors the FIRST `set_tracer_provider` call per process —
    subsequent ones log a warning and are ignored. So we set once per session
    and clear the exporter's buffer between tests instead of re-wiring.

    Each test stacks one extra `SimpleSpanProcessor` onto the same provider.
    Practically harmless at this suite size (each exporter buffers
    independently); if the suite grows materially, switch to a session-scoped
    provider+processor with function-scoped exporters and a teardown that
    detaches the processor.
    """
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    yield exp
    exp.clear()


def _call(tool_call_id: str = "tc1", name: str = "get_weather") -> ToolCall:
    return ToolCall(tool_call_id=tool_call_id, name=name, arguments={})


def test_on_seal_emits_tool_sealed_span(exporter: InMemorySpanExporter) -> None:
    obs = OTelObservability()
    obs.on_seal(
        SealEvent(
            kind="tool_sealed",
            tool_call=_call("tc1", "get_weather"),
            seal_latency_ms=12.5,
        )
    )

    spans = [s for s in exporter.get_finished_spans() if s.name == "eager_tools.seal"]
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs["eager.seal.kind"] == "tool_sealed"
    assert attrs["eager.seal.latency_ms"] == 12.5
    assert attrs["eager.tool.id"] == "tc1"
    assert attrs["eager.tool.name"] == "get_weather"


def test_on_seal_message_complete_span(exporter: InMemorySpanExporter) -> None:
    obs = OTelObservability()
    obs.on_seal(SealEvent(kind="message_complete"))

    spans = [s for s in exporter.get_finished_spans() if s.name == "eager_tools.seal"]
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs["eager.seal.kind"] == "message_complete"
    assert "eager.tool.id" not in attrs
    assert "eager.seal.latency_ms" not in attrs


def test_on_seal_parse_error_records_exception(exporter: InMemorySpanExporter) -> None:
    obs = OTelObservability()
    err = ValueError("malformed JSON")
    obs.on_seal(
        SealEvent(
            kind="tool_sealed",
            tool_call=_call("tc-bad", "get_weather"),
            parse_error=err,
        )
    )

    spans = [s for s in exporter.get_finished_spans() if s.name == "eager_tools.seal"]
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert "ValueError" in attrs["eager.parse_error"]
    # record_exception adds an "exception" event to the span.
    event_names = [ev.name for ev in spans[0].events]
    assert "exception" in event_names
    # No dispatch span fires for parse-error — adapter calls record_error
    # which bypasses dispatch hooks. Asserted by absence here.
    assert not [s for s in exporter.get_finished_spans() if s.name == "eager_tools.dispatch"]


def test_dispatch_lifecycle_emits_paired_span(exporter: InMemorySpanExporter) -> None:
    obs = OTelObservability()
    call = _call("tc2", "get_news")
    obs.on_dispatch_start(call)
    obs.on_dispatch_end(call, None)

    spans = [s for s in exporter.get_finished_spans() if s.name == "eager_tools.dispatch"]
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs["eager.tool.id"] == "tc2"
    assert attrs["eager.tool.name"] == "get_news"
    assert "eager.error.type" not in attrs


def test_dispatch_end_with_error_records_error(exporter: InMemorySpanExporter) -> None:
    obs = OTelObservability()
    call = _call("tc3", "broken_tool")
    obs.on_dispatch_start(call)
    obs.on_dispatch_end(call, RuntimeError("kaboom"))

    spans = [s for s in exporter.get_finished_spans() if s.name == "eager_tools.dispatch"]
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs["eager.error.type"] == "RuntimeError"
    event_names = [ev.name for ev in spans[0].events]
    assert "exception" in event_names


def test_init_raises_clear_error_when_otel_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construct under a poisoned `opentelemetry` import, expect a hint-bearing ImportError.

    Mechanism: Python's import system treats `sys.modules[name] = None` as a
    poisoned entry — any subsequent `import name` raises `ModuleNotFoundError`
    (a subclass of `ImportError`). Standard pytest pattern but worth flagging
    in case Python's import semantics change in some future release.
    """
    # Remove cached subpackages so the fresh `from opentelemetry import trace` re-imports.
    for mod in list(sys.modules):
        if mod == "opentelemetry" or mod.startswith("opentelemetry."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setitem(sys.modules, "opentelemetry", None)

    with pytest.raises(ImportError, match=r"\[otel\] extra"):
        OTelObservability()
