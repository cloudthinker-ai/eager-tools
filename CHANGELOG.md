# Changelog

All notable changes to this repo are tracked here. Per-package versions move
in lockstep with the relevant package release.

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
