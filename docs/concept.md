# Concept — Eager Tool Calling in 90 seconds

> **TL;DR.** The runtime dispatches each tool the moment its streaming JSON
> block ends — without waiting for the rest of the assistant message to
> finish. Tool execution overlaps with model generation, not just with other
> tools. Wall clock collapses from `sum(tools)` to `max(stream, max(tool))`.

For the full canonical reference (invariants, edge cases, contract clauses),
see [`METHOD.md`](../METHOD.md).

---

## The three eras

| Era | Concurrency | When tools start | Wall clock |
|-----|-------------|------------------|------------|
| Sequential | none | after each prior tool | `Σ(stream + all tools)` |
| Parallel   | tools with tools | after `message_stop` | `stream + max(tool)` |
| **Eager**  | tools with tools **and** tools with generation | the instant each tool's block ends | **`max(stream, max(tool))`** |

Sequential → parallel collapses one dimension of latency. Parallel → eager
collapses the other. Time goes from sum to max in both directions.

---

## The mechanism

A streaming response from Claude or GPT arrives as chunks. Each `tool_use`
block has a stable `tool_call_id`. The first chunk of a tool carries the id +
name; subsequent chunks carry only argument JSON fragments routed by `index`.

> **Invariant.** The instant a chunk arrives with a `tool_call_id` different
> from the previous one, the prior tool is definitionally complete. Its
> arguments are fully accumulated. It is safe to execute *right now*, even
> though the assistant message is still streaming.

This is the **seal event**. The runtime detects it, hands the sealed
`ToolCall` to an async executor pool, and keeps reading the stream.

### Chunk-by-chunk trace

```
chunk(id = A, args = "{path:")    → accumulate into buffer[A]
chunk(id = A, args = "/etc}")     → accumulate
chunk(id = B, args = "{url:")     → NEW id → SEAL A → dispatch tool A
chunk(id = B, args = "...")       → accumulate B     (A executing in background)
chunk(id = C, args = "{query:")   → SEAL B → dispatch tool B
                                  → (A, B executing while C still streaming)
message_stop                      → SEAL C → dispatch tool C
                                  → (A, B, C all overlap)
```

### Lane-overlap timeline

```
stream : [==================================]
tool A :   [=========]            ← fires when B's id arrives
tool B :       [=========]        ← fires when C's id arrives; runs concurrent with A
tool C :           [=========]    ← fires at message_stop; runs concurrent with A + B
```

**Correctness test:** drop a vertical line at the 50% mark. It must cut
through all three bars. If tool bars are end-to-end instead of overlapping,
the implementation is sequential-in-disguise, not eager.

---

## Mental model

Eager tool calling is **CPU instruction pipelining for agents**. A modern CPU
does not wait for one instruction to retire before decoding the next — it
fills the pipeline, hides latency, and runs multiple stages concurrently. The
agent runtime is the CPU. The model stream is instruction fetch. The tools
are execution units. The seal event is the register-ready signal that lets
execution begin before the rest of the batch is decoded.

Once that analogy clicks, the implementation falls out of it: a streaming
parser, a seal detector, an async executor pool, per-block cancellation, and
idempotency gating. Everything else is bookkeeping.

---

## See also

- [`METHOD.md`](../METHOD.md) — full method reference, runtime contract, edge cases
- [`when-not-to-use.md`](when-not-to-use.md) — when classic dispatch is the right answer
- [`rfc-streaming-tool-dispatch-protocol.md`](rfc-streaming-tool-dispatch-protocol.md) — proposal for a cross-provider standard
- [`../examples/01_minimal.py`](../examples/01_minimal.py) — runnable demo
