# Changelog

All notable changes to this repo are tracked here. Per-package versions move
in lockstep with the relevant package release.

## 0.2.0 — per-call gate (HITL primitive)

Releases `eager-tools-core`, `eager-tools-anthropic`, `eager-tools-openai`,
and `eager-tools-langgraph` together at 0.2.0.

### Added

- **Optional `Tool.gate: GateFn` attribute** — per-call gate evaluated after
  the JSON block seals (parsed args visible). Returning `False` (or
  raising) routes the call off the eager path; the call still appears on the
  AIMessage so the framework's tool step can run it. Composes with framework
  HITL primitives (LangGraph `interrupt()`, `HumanInTheLoopMiddleware`).
  See [`docs/hitl.md`](./docs/hitl.md).
- **`GateFn`** public type alias (`Callable[[ToolCall], Awaitable[bool]]`)
  exported from `eager_tools` for user annotations.
- **`EagerDispatchDeniedError`** exception base, with `NonIdempotentToolError`
  and new `GateDeniedError` as subclasses. Adapters catch the base.
- **`examples/09_gate.py`** — minimal runnable demo of a path-allowlist gate.

### Changed (potentially breaking for custom error-handling code)

- **`NonIdempotentToolError` is now a subclass of `EagerDispatchDeniedError`.**
  Code that catches `NonIdempotentToolError` continues to work for
  idempotency denial. To catch BOTH idempotency and gate denial uniformly,
  migrate to `except EagerDispatchDeniedError`. The bundled LangGraph adapter is
  migrated in this release; users with custom adapters that catch
  `NonIdempotentToolError` need to update.
- **Anthropic / OpenAI adapters no longer crash on non-idempotent tool
  registration.** Previously `_handle_seal` called `pool.dispatch(...)`
  with no try/except — registering a non-idempotent tool would raise
  mid-stream. Now the adapter records the denial in `results()` (as the
  original cause when the gate raised, or `GateDeniedError` /
  `NonIdempotentToolError` otherwise) and the stream continues. Latent
  bug fix.

### Notes

- **Critical-path contract**: gates are awaited inside the
  stream-consumption task. A slow gate halts subsequent chunk reads and
  may trip provider keepalive timeouts. Locked in by
  `test_slow_gate_blocks_stream_progress` in both adapter suites; if that
  test ever passes accidentally (e.g. someone "fixed" the blocking by
  decoupling gate evaluation from the stream), the eager invariant is
  broken. Slow approval flows belong at the framework layer, not in the
  gate. See [`docs/hitl.md`](./docs/hitl.md) for the full guidance.
- The gate is duck-typed (`getattr(tool, "gate", None)`), not declared on
  the `Tool` Protocol — adding `gate` to the Protocol would have broken
  `isinstance(my_tool, Tool)` for every existing user tool. `GateFn` is
  exported as a public type alias to give users something to annotate
  against.
- No `on_dispatch_denied` observability hook in this release. Sealed-but-
  never-dispatched calls are inferable by comparing seal events to
  `on_dispatch_start` calls; revisit if dashboards need direct counters.

## eager-tools-core 0.1.0 — OTel hooks

### Added

- `OTelObservability` reference implementation of `ObservabilityHook`, exported
  from the top-level `eager_tools` package. Opt-in via the `[otel]` extra
  (`pip install eager-tools-core[otel]`); core remains zero-dep at runtime.
  Span schema documented in `observability.py` — `eager_tools.seal` (sync) and
  `eager_tools.dispatch` (lifecycle), namespaced under `eager.*` attributes.

### Changed (potentially breaking for custom `ObservabilityHook` impls)

- **`on_seal()` now fires.** Previously the protocol method existed but no
  adapter called it. Now every adapter (`eager-tools-anthropic`,
  `eager-tools-openai`, `eager-tools-langgraph`) invokes
  `observability.on_seal(seal)` once per sealed tool plus once for
  `message_complete` (where applicable — the LangGraph middleware does not
  synthesize `message_complete` events). Implementations that previously
  ignored or stubbed `on_seal` will start receiving calls. The adapters wrap
  each invocation in `contextlib.suppress(Exception)` so observer bugs cannot
  break a stream — same semantics as `ExecutorPool._safe_hook`.

### Notes

- `OTelObservability` does not close dispatch spans for tool tasks cancelled
  while queued on the executor's semaphore. Bounded leak (≤ N pending tools
  per cancelled stream); documented in the class docstring. Will revisit if
  real users hit it.
- Parse-error seals (`SealEvent.parse_error is not None`) emit a `seal` span
  but no paired `dispatch` span — the tool was malformed and never executed.
  This is intentional; `ExecutorPool.record_error` bypasses dispatch hooks.
