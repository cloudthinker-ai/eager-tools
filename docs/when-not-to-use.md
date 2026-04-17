# When NOT to use eager tool calling

Eager dispatch is a wall-clock optimization, not a free upgrade. Several
shapes of work are either incompatible or actively hurt by it. If your
situation matches any below, **use the classic post-`message_stop` dispatch
path instead** — it stays a one-line decision per tool via the
`idempotent: bool` flag, or per-app by skipping the adapter entirely.

---

## 1. Non-idempotent tools

The model occasionally emits a tool call, reconsiders, and replaces or
retracts it before `message_stop`. A tool that already fired eagerly cannot
be un-fired. Anything with side effects is at risk:

- payments, refunds, ledger writes
- outbound messages (email, Slack, SMS)
- `rm`, `DELETE`, `DROP TABLE`, force-pushes
- queue enqueues, webhook fires
- destructive admin operations

**What to do.** Mark the tool `idempotent = False`. The `ExecutorPool`
rejects it via `NonIdempotentToolError` so the caller routes it to the
classic path that fires only after `message_stop`, when the model has
committed.

---

## 2. Sequential tool dependencies

If tool B's *arguments* require tool A's *result*, the model cannot emit B
until A returns. There is no pipeline opportunity — eager and classic paths
degrade to identical wall-clock behavior. This is a natural ceiling, not a
bug.

**What to do.** Run as usual. The runtime will simply find no overlap; nothing
is broken or slower. If most of your workload looks like this, the eager
adapter is not the right place to invest.

---

## 3. Tools faster than the network round-trip

If your tool is a few milliseconds (an in-memory cache lookup, a local
computation), the dispatch / seal-detection / cross-task overhead can rival
the tool itself. You will not see a measurable win.

**What to do.** Keep eager dispatch on for the slow tools (network I/O,
LLM calls, DB queries) and trust the runtime to ignore the fast ones — they
finish before they bottleneck anything. If *every* tool is sub-millisecond,
the classic path is genuinely simpler.

---

## 4. Non-streaming providers or buffered gateways

Sealing requires per-chunk visibility. If the provider, gateway, or proxy
buffers the full response before forwarding (some enterprise LLM gateways
do this for content moderation), there is nothing to seal mid-stream.

**What to do.** Detect non-streaming responses and fall back. The adapter
constructor takes any `AsyncIterator` — supply one that yields a single
final chunk and the runtime degrades to the classic path naturally.

---

## 5. Single-tool conversations

If the model only ever calls one tool per turn, there is nothing to overlap
*with*. Eager dispatch fires the tool the moment the block ends instead of
at `message_stop`, which saves only the trailing-text generation time —
sometimes worth it, often noise.

**What to do.** Measure. If turns are usually one-tool, the eager adapter is
still safe but the win shrinks. Multi-tool conversations are where eager
shines.

---

## 6. Non-async codebases

`eager-tools-core` is async-only by design. Async-only halves our test
matrix, matches modern provider SDKs (`AsyncAnthropic`, `AsyncOpenAI`), and
keeps the cancellation story coherent. There is no sync wrapper and we do
not plan to add one.

**What to do.** Run eager dispatch in an isolated async loop
(`asyncio.run(...)` per request) if you must integrate from a sync host. Or
stay on the classic post-`message_stop` path until you migrate.

---

## Summary table

| Situation | Use eager? | Why |
|-----------|------------|-----|
| Idempotent slow tool (HTTP GET, read-only DB query) | ✅ | Wins are largest here. |
| Side-effecting tool (payment, email, delete) | ❌ | Mark `idempotent=False`; classic path. |
| B depends on A's result | ⚠️ degrades to classic | No pipeline available; eager is harmless. |
| Sub-ms tool | ⚠️ negligible win | Adapter overhead may dominate. |
| Buffered / non-streaming provider | ❌ | Nothing to seal. |
| One-tool conversations | ⚠️ minor win | Only saves trailing-text time. |
| Sync-only codebase | ❌ | Eager-tools is async-only. |

If your situation matches a ❌ row, use the classic post-`message_stop`
dispatch path instead. The eager runtime and classic dispatch coexist
cleanly per tool — eager is always opt-in, never destructive when omitted.
