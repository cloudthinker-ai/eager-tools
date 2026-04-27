# Changelog

All notable changes to this repo are tracked here. Per-package versions move
in lockstep with the relevant package release.

## 0.3.0 — gate reasons + denial observability + OTel cancel-leak fix

Releases `eager-tools-core`, `eager-tools-anthropic`, `eager-tools-openai`,
and `eager-tools-langgraph` together at 0.3.0.

### Added

- **Gate can return `bool | str`.** Returning a string denies eager dispatch
  and uses that string as the denial reason. The string is exposed on
  `GateDeniedError.reason`, becomes the exception message verbatim, and
  flows through `on_dispatch_denied` — adapters that surface the denial
  back to the model now send the gate's own explanation instead of a
  wrapper. `GateFn` alias updated to `Callable[[ToolCall], Awaitable[bool | str]]`.
  Empty string `""` is still a denial (the type, not truthiness, decides).
- **`ObservabilityHook.on_dispatch_denied(call, reason: str)`** — fired
  immediately before each `EagerDispatchDeniedError` is raised. Four
  denial paths, each with its own `reason`:
  - `"non-idempotent tool"`              — `tool.idempotent is False`
  - the gate's `str` verbatim (incl. `""`) — gate returned a string
  - `"gate denied <name>"`               — gate returned `False`
  - `"gate raised <Type>: <msg>"`        — gate raised an exception

  Unpaired with `on_dispatch_start` / `on_dispatch_end` — the tool never
  reached the executor. `OTelObservability` emits a synchronous
  `eager_tools.dispatch_denied` span with `eager.tool.id`, `eager.tool.name`,
  `eager.denial.reason`.
- **`reason: str` attribute on `EagerDispatchDeniedError`** (and both
  subclasses). Adapters can `exc.reason` instead of parsing `str(exc)`.

### Fixed

- **OTel dispatch span no longer leaks on cancel-while-queued.**
  Pre-0.3.0, tool tasks cancelled while waiting on the pool's semaphore
  never reached `on_dispatch_end`, leaving the corresponding
  `eager_tools.dispatch` span unclosed (bounded but real — known
  limitation #2 in `observability.py`). `_run_one` now uses a single
  `try/finally`, so `on_dispatch_end` fires on every exit path. The
  limitation note has been dropped from `observability.py`. Pinned by
  `test_dispatch_end_fires_on_cancel_while_queued`.

### Changed (potentially breaking for custom `ObservabilityHook` impls)

- **`ObservabilityHook` Protocol now declares `on_dispatch_denied`.** Custom
  impls written against the 0.2.x Protocol will:
  - Continue to work at runtime — the executor wraps every hook call in
    `contextlib.suppress(Exception)`, which catches the `AttributeError`
    when the method is missing. The denial events are silently dropped.
  - Fail `isinstance(x, ObservabilityHook)` because the Protocol is
    `runtime_checkable`. Add a no-op `def on_dispatch_denied(self, call, reason): pass`
    to restore conformance.

  Same migration shape as the v0.1 `on_seal` Protocol expansion.

- **`EagerDispatchDeniedError` constructor now requires `reason: str` keyword.**
  Custom adapters that subclassed or instantiated `NonIdempotentToolError` /
  `GateDeniedError` directly need to add `reason="..."`. The bundled
  adapters do not — they only catch the exceptions, never construct them.

### Notes

- The `bool | str` change is fully additive at the gate-return-type level —
  existing `True`/`False` gates work unchanged. The new `str` path is
  opt-in.
- `on_dispatch_denied` fires from the executor (not the adapter), so it's
  consistent across Anthropic / OpenAI / LangGraph regardless of how each
  adapter routes the denial downstream.
- Limitation note dropped: parse-error seals still emit a `seal` span
  without a paired `dispatch` span (limitation #1, intentional). That
  remains.

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
