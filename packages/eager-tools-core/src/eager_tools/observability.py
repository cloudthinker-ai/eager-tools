# pyright: reportMissingImports=false
"""Optional OpenTelemetry implementation of `ObservabilityHook`.

Opt-in: requires the `[otel]` extra (`pip install eager-tools-core[otel]`).
Importing this module does NOT pull `opentelemetry` into your runtime — the
import is deferred to `OTelObservability.__init__`. The core package keeps its
zero-dep promise.

Span schema
-----------
Three span types — the dispatch span is long-lived, the others sync.

    eager_tools.seal           — sync. Tool block seals (or message ends).
                                 Carries `eager.seal.kind`,
                                 `eager.seal.latency_ms`, `eager.tool.id`,
                                 `eager.tool.name`. On parse_error seals:
                                 `eager.parse_error` + `record_exception`.

    eager_tools.dispatch       — opened on `on_dispatch_start`, closed on
                                 `on_dispatch_end`. Carries `eager.tool.id`,
                                 `eager.tool.name`. On error:
                                 `record_exception` + `eager.error.type`.

    eager_tools.dispatch_denied — sync. Sealed tool denied eager dispatch
                                  (non-idempotent / gate False / gate str /
                                  gate raised). Carries `eager.tool.id`,
                                  `eager.tool.name`, `eager.denial.reason`.
                                  No paired dispatch span — the tool never
                                  reached the executor.

Attribute namespace `eager.*` avoids collision with upstream provider
instrumentations (Anthropic, OpenAI, LangChain SDKs).

Known limitations
-----------------
1. Parse-error seals do NOT have a paired dispatch span. The tool was malformed
   and never executed — `ExecutorPool.record_error` bypasses dispatch hooks
   entirely. This truthfully reflects "seen but not run" in dashboards.

2. `on_seal` for `kind="message_complete"` fires a span with only the kind
   attribute set — symmetry with sealed-tool spans is more useful for
   dashboards than terseness.

PII / data exposure
-------------------
Parse-error seal spans set `eager.parse_error = repr(parse_error)`, and
`record_exception(parse_error)` attaches the full exception. For
`json.JSONDecodeError` and friends, that includes the offending JSON
verbatim. If your tools accept user-derived content, treat the resulting
traces as PII-bearing — they may surface in whichever backend the configured
exporter ships to (Datadog, Honeycomb, Langfuse, etc.). Scrub or sample
accordingly at the SDK / collector level.
"""

from __future__ import annotations

from typing import Any

from .types import SealEvent, ToolCall


class OTelObservability:
    """OpenTelemetry-backed `ObservabilityHook`.

    Pass to any adapter that takes `observability=`:

        from eager_tools import OTelObservability
        from eager_tools_anthropic import AnthropicEagerStream

        runner = AnthropicEagerStream(
            source, tools=my_tools,
            observability=OTelObservability(),
        )

    Requires an OTel `TracerProvider` to be configured globally; this class
    only emits spans and does not own pipeline / exporter setup. See the
    `opentelemetry-sdk` docs for `TracerProvider` + exporter wiring.

    Threading: single-event-loop only. `_dispatch_spans` is a plain dict, not
    a `threading.Lock`-protected one — safe under asyncio (cooperative
    single-thread), torn under `asyncio.to_thread` / `concurrent.futures`. If
    you need cross-thread observability, wrap this instance behind your own
    serialization or instantiate one per loop.
    """

    def __init__(self, tracer_name: str = "eager_tools") -> None:
        try:
            from opentelemetry import trace
        except ImportError as exc:
            raise ImportError(
                "OTelObservability requires the [otel] extra; install eager-tools-core[otel]"
            ) from exc
        self._tracer = trace.get_tracer(tracer_name)
        self._dispatch_spans: dict[str, Any] = {}

    def on_seal(self, event: SealEvent) -> None:
        with self._tracer.start_as_current_span("eager_tools.seal") as span:
            span.set_attribute("eager.seal.kind", event.kind)
            if event.seal_latency_ms is not None:
                span.set_attribute("eager.seal.latency_ms", event.seal_latency_ms)
            if event.tool_call is not None:
                span.set_attribute("eager.tool.id", event.tool_call.tool_call_id)
                span.set_attribute("eager.tool.name", event.tool_call.name)
            if event.parse_error is not None:
                span.set_attribute("eager.parse_error", repr(event.parse_error))
                span.record_exception(event.parse_error)

    def on_dispatch_start(self, call: ToolCall) -> None:
        span = self._tracer.start_span("eager_tools.dispatch")
        span.set_attribute("eager.tool.id", call.tool_call_id)
        span.set_attribute("eager.tool.name", call.name)
        self._dispatch_spans[call.tool_call_id] = span

    def on_dispatch_end(self, call: ToolCall, error: Exception | None) -> None:
        span = self._dispatch_spans.pop(call.tool_call_id, None)
        if span is None:
            return
        if error is not None:
            span.set_attribute("eager.error.type", type(error).__name__)
            span.record_exception(error)
        span.end()

    def on_dispatch_denied(self, call: ToolCall, reason: str) -> None:
        with self._tracer.start_as_current_span("eager_tools.dispatch_denied") as span:
            span.set_attribute("eager.tool.id", call.tool_call_id)
            span.set_attribute("eager.tool.name", call.name)
            span.set_attribute("eager.denial.reason", reason)


__all__ = ["OTelObservability"]
