# Human-in-the-loop with eager tools

Eager-tools provides a **per-call gate** at seal time. Approval flow,
queueing, edit-and-rerun — those live at the agent-framework layer
(LangGraph `interrupt()`, OpenAI Agents SDK `needsApproval`, your own
review queue). The two compose: the gate decides whether *eager* dispatch
is allowed; the framework's tool step handles whatever the gate denied.

This split is deliberate. Replicating the full {approve, edit, reject,
respond} decision space in the library would create two competing HITL
APIs in users' codebases. The gate answers one question — "fire eagerly:
yes/no" — and gets out of the way.

## The two knobs

| Knob | Granularity | When to use |
|------|-------------|-------------|
| `Tool.idempotent: bool` | per-tool, blanket | The whole tool is unsafe to fire before `message_stop` (payments, deletes). |
| `Tool.gate: GateFn`     | per-call, args-aware | Sometimes safe, sometimes not (path allowlist, query type, tenant policy). |

A non-idempotent tool is *never* dispatched eagerly. A gated tool is
dispatched iff the gate returns `True`. The two stack — the idempotency
check runs first, then the gate.

## The gate signature

```python
from eager_tools import GateFn, ToolCall

# GateFn = Callable[[ToolCall], Awaitable[bool | str]]
# - True   → allow
# - False  → deny (no reason)
# - str    → deny with that string as the reason. The string flows to
#            on_dispatch_denied AND becomes the GateDeniedError message —
#            so adapters that surface the denial to the model send a useful
#            explanation, not a wrapper. Empty string `""` is still a denial.

class ReadFile:
    name = "read_file"
    idempotent = True

    async def gate(self, call: ToolCall) -> bool | str:
        path = call.arguments.get("path", "")
        if path.startswith("/etc/"):
            return f"path {path!r} is in the system-config denylist"
        return True

    async def __call__(self, arguments: dict[str, Any]) -> Any: ...
```

Note the `.get("path", "")` — gates are responsible for handling missing or
malformed args defensively. A gate that raises (e.g. `KeyError`) is treated
as denial: `GateDeniedError` is raised with the original exception on
`__cause__` and `reason="gate raised <Type>: <msg>"`.

The runtime reads `getattr(tool, "gate", None)`. The attribute is
duck-typed — it is **not** declared on the `Tool` Protocol so existing
tools without a gate stay backward-compatible.

`call.conversation_id` is populated when the adapter was constructed with
`conversation_id=...`, useful for multi-tenant policy lookups.

## ⚠ Critical-path warning: keep gates sync-fast

The gate is awaited inside the same task that consumes the LLM stream. A
slow gate **starves the stream** — the next tool block cannot seal until
the gate returns, and a sufficiently slow gate will trip the provider's
keepalive timeout and kill the connection.

**Sync-fast** means: in-memory predicates, allowlists/denylists, cached
policy decisions (OPA / Cedar with a local cache), tenant flags. Anything
sub-millisecond.

**NOT sync-fast** — and therefore must NOT live in a gate:

- `await ask_human_for_approval(call)`
- `await opa.evaluate(...)` against a remote service
- `await db.query(...)` for fresh policy state on every call

Slow approval flows belong at the framework layer, not in the gate.

## Composition recipe — LangGraph

```python
from eager_tools import ToolCall
from eager_tools_langgraph import EagerMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

class ReadFile:
    name = "read_file"
    idempotent = True

    async def gate(self, call: ToolCall) -> bool | str:
        path = call.arguments.get("path", "")
        if path.startswith("/etc/"):
            return f"path {path!r} is in the system-config denylist"
        return True

    async def __call__(self, arguments): ...

agent = create_agent(
    model=...,
    tools=[ReadFile()],
    middleware=[
        EagerMiddleware({"read_file": ReadFile()}),
        HumanInTheLoopMiddleware(...),  # handles the slow approval step
    ],
)
```

When the gate returns `False`, `EagerMiddleware` skips eager dispatch
silently. The AIMessage still carries the gated `tool_call` so the agent's
normal tool step picks it up — and `HumanInTheLoopMiddleware` /
`interrupt()` runs the human-wait at *that* layer, where the stream is no
longer in flight.

This is the headline pattern: **fast policy at seal time, slow approval at
the tool step.**

## Composition recipe — direct Anthropic / OpenAI

If you're driving the stream yourself rather than going through LangGraph:

```python
from eager_tools import GateDeniedError
from eager_tools_anthropic import AnthropicEagerStream

stream = AnthropicEagerStream(source, tools={"read_file": ReadFile()})

async for ev in stream.events():
    ...

async for call, payload in stream.results():
    if isinstance(payload, GateDeniedError):
        # Route to your own approval queue, synthesize a "denied" tool message,
        # log for audit, etc. The original gate exception (if any) is on
        # payload.__cause__ when the gate raised.
        await approval_queue.put(call)
    elif isinstance(payload, BaseException):
        ...
    else:
        ...
```

Gate-denied calls surface in `results()` as the **original gate
exception** when the gate raised, or as `GateDeniedError` when the gate
returned `False`. The adapter unwraps `__cause__` automatically so user
code sees the meaningful exception, not the wrapper.

## Cancellation contract

If the stream is cancelled while a gate is awaiting (user interrupt,
provider disconnect, timeout), the gate task receives `CancelledError`
and propagates per asyncio semantics — the same contract as tool
execution.

Gates with side effects (queue inserts, audit logs, lock acquisition)
must clean up in their own `try/finally`:

```python
async def gate(self, call: ToolCall) -> bool:
    await self._audit_log.append(call)
    try:
        decision = self._policy.evaluate(call)  # sync-fast
        return decision.allowed
    finally:
        await self._audit_log.flush()
```

## Error hierarchy

```
RuntimeError
└── EagerDispatchDeniedError      # catch-all for "must not fire eagerly"
    ├── NonIdempotentToolError   # idempotent=False
    └── GateDeniedError          # gate returned False / str / raised
```

Both subclasses carry a `reason: str` attribute. Adapters that route
denials back to the model can use it directly:

```python
try:
    await pool.dispatch(call)
except EagerDispatchDeniedError as exc:
    tool_error_message = exc.reason  # safe to send to the LLM
```

Adapters catch `EagerDispatchDeniedError` to handle both cases uniformly. Code
that previously caught only `NonIdempotentToolError` continues to work for
idempotency denial, but will miss gate denial — migrate to
`EagerDispatchDeniedError` to catch both.

## Observability

Every denial fires `ObservabilityHook.on_dispatch_denied(call, reason)`
*before* the exception is raised — so traces and counters stay accurate
regardless of how the adapter handles the denial. Three reason shapes:

| Path | `reason` value |
|------|----------------|
| `idempotent=False` | `"non-idempotent tool"` |
| Gate returned `str` | The string verbatim (including `""`) |
| Gate returned `False` | `"gate denied <name>"` |
| Gate raised | `"gate raised <Type>: <msg>"` |

These calls are unpaired — there is **no** matching `on_dispatch_start` /
`on_dispatch_end`, because the tool never reached the executor. The OTel
backend emits a synchronous `eager_tools.dispatch_denied` span per denial.

## What eager-tools is NOT trying to be

- **Not an approval UI.** Build it (or use your framework's primitive).
- **Not an edit-then-run API.** If you need to mutate args before running,
  consume the call from `results()` (or the AIMessage's `tool_calls` in
  LangGraph) and re-dispatch externally.
- **Not a "sometimes eager for non-idempotent tools" knob.** A gate cannot
  flip `idempotent=False` to "sometimes eager" — that would violate the
  invariant that non-idempotent tools never race the stream (which exists
  because the model can retract them mid-stream).

## Further reading

- [`docs/when-not-to-use.md`](./when-not-to-use.md) — broader fit guide.
- [`METHOD.md`](../METHOD.md) — provider-agnostic mechanism reference.
- [`examples/09_gate.py`](../examples/09_gate.py) — minimal runnable demo.
