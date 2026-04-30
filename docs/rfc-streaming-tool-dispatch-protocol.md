# RFC — Streaming Tool Dispatch Protocol (STDP)

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Version** | 0.1 |
| **Authors** | CloudThinker (eager-tools maintainers) |
| **Date** | 2026-04-15 |
| **Target** | LLM provider streaming APIs (SSE / HTTP/2 streaming) |
| **Relates to** | `METHOD.md` in this repo, the [`eager-tools`](../README.md) reference implementation |

> **Call for review.** This RFC proposes a minimal, provider-neutral event shape that enables **eager tool dispatch** — tools starting execution the moment their block finishes streaming, without waiting for the full assistant message. Feedback welcome via [GitHub Discussions](https://github.com/cloudthinker-ai/eager-tools/discussions).

---

## 1. Abstract

This document specifies the **Streaming Tool Dispatch Protocol (STDP)** — a minimal, backwards-compatible extension to existing LLM streaming APIs that enables runtimes to dispatch each tool call the instant its streaming block finishes, overlapping tool execution with ongoing LLM generation.

STDP requires providers to emit a `tool_block_complete` event (or equivalent signal) at a deterministic point in the stream. It does not change the shape of existing `tool_use` content blocks, the order of events, or the semantics of `message_stop`. Implementations that ignore the new event continue to function unchanged.

STDP is designed to be implementable by every major LLM provider (Anthropic, OpenAI, Google, Amazon Bedrock, Mistral, local models served via vLLM/TGI) without breaking existing consumers.

---

## 2. Motivation

### 2.1 The latency that isn't being measured

A typical agent turn today:

```
stream phase :  [============================]       (~3–5 seconds)
tool phase   :                                [=====] (~2–10 seconds)
wall clock   :  [==================================]
```

Parallel tool calling — standard across major providers — reduced the tool phase from Σ(tools) to max(tool). Real, measurable progress. But the stream phase still sits in front of the tool phase as serial dead weight.

On tool-heavy workloads (cost audits, multi-step debugging, search-and-summarize), the stream and tool phases are roughly equal. Removing the serialization between them halves wall clock. Production data from CloudThinker shows **1.5× to 21× speedups** depending on workload.

### 2.2 Why runtimes can't solve this alone today

Existing streaming APIs expose tool_use blocks as a sequence of `content_block_start` / `content_block_delta` / `content_block_stop` events (Anthropic) or cumulative `tool_calls` array deltas (OpenAI). Runtimes *can* detect per-block completion by:

- **Anthropic**: listening for `content_block_stop` events where the block type was `tool_use`.
- **OpenAI**: comparing consecutive `tool_calls[i]` shapes and inferring that a shape is complete when no further deltas arrive. This is **heuristic and fragile** — the provider makes no guarantee that delta ordering implies completion.

The result: every runtime invents its own ad-hoc seal detector. Cross-provider correctness cannot be verified from the wire. A standardized event removes the guesswork and lets providers optimize their own buffering without silently breaking consumers.

### 2.3 What STDP provides

- A **single, unambiguous, forward-compatible** signal that a tool call block is fully streamed.
- A **negotiated opt-in** — consumers indicate STDP support in request headers; providers emit the event only when requested.
- A **reference implementation** ([`eager-tools`](../README.md)) that demonstrates correct behavior across Anthropic and OpenAI today and serves as the conformance test suite for future providers.

---

## 3. Terminology

The key words *MUST*, *MUST NOT*, *REQUIRED*, *SHALL*, *SHALL NOT*, *SHOULD*, *SHOULD NOT*, *RECOMMENDED*, *MAY*, and *OPTIONAL* in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

- **Tool block**: A region of the streaming response representing a single tool invocation (Anthropic: a `tool_use` content block; OpenAI: one entry in the `tool_calls` array).
- **Seal event**: The moment a tool block is definitionally complete — its `name` and `arguments` will not receive further deltas.
- **Runtime**: The consumer of the LLM stream (agent framework, application code). The entity that dispatches tools.
- **Provider**: The entity emitting the stream (Anthropic, OpenAI, etc.).
- **Eager dispatch**: Executing a tool call after its seal event but before `message_stop`.

---

## 4. Protocol Specification

### 4.1 Negotiation

Runtimes opt into STDP via a request header:

```
X-Streaming-Tool-Dispatch: v1
```

Providers that do not support STDP respond normally (no new events emitted). Runtimes detect support by the presence or absence of the events specified below — no capability-discovery endpoint is required in v0.1.

Alternative: providers MAY expose STDP support via a field on the existing model-capability endpoint (`/v1/models/{id}`). This is not normative in v0.1.

### 4.2 New event — `tool_block_complete`

Providers that support STDP MUST emit a `tool_block_complete` event exactly once per tool block, immediately after the last delta for that block and before any content from a subsequent block.

```json
event: tool_block_complete
data: {
  "tool_call_id": "toolu_abc123",
  "index": 2,
  "name": "read_file",
  "arguments": {"path": "/etc/hosts"},
  "sealed_at_ms": 1420
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tool_call_id` | string | yes | Provider-assigned stable identifier for this tool call. Same value used in the eventual `tool_result`. |
| `index` | integer | yes | Zero-based block index within the assistant message. Enables routing of out-of-order late deltas (see §5.3). |
| `name` | string | yes | The tool name the model selected. |
| `arguments` | object | yes | Fully-parsed arguments. MUST be valid JSON; providers MUST NOT emit this event with partial arguments. |
| `sealed_at_ms` | integer | no | Provider-side wall-clock offset from request start, milliseconds. For observability. |

### 4.3 Invariants

- **(I1)** `tool_block_complete` for block N MUST appear in the stream before any `content_block_delta` for block N+1.
- **(I2)** Every `tool_use` block in the assistant message MUST be accompanied by exactly one `tool_block_complete` event.
- **(I3)** `tool_block_complete` MUST precede `message_stop` for each in-flight block.
- **(I4)** Providers MUST NOT retroactively re-emit `tool_block_complete` for a block that has already been sealed.

### 4.4 Interaction with existing events

STDP is purely additive. Providers continue to emit `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop` per their existing streaming specifications. Runtimes that ignore `tool_block_complete` observe identical behavior to a non-STDP stream.

### 4.5 Retraction

The model MAY, in rare cases, retract a tool call after emitting it (correcting a mistake mid-stream). STDP does NOT add a retraction event in v0.1. Runtimes MUST handle retraction by treating `Tool.idempotent == false` as non-eager: non-idempotent tools are dispatched only after `message_stop`, preserving retraction safety.

A future STDP revision MAY add `tool_block_retracted` if field deployment data justifies it.

---

## 5. Reference Behavior

### 5.1 Runtime state machine

A conforming runtime maintains, per conversation:

```
state = {
  last_sealed_id: string | null,     # most recent sealed tool_call_id
  buffers: {                          # accumulated block state, keyed by id
    [tool_call_id]: {
      name: string | null,
      args_fragments: list[string],
    }
  },
  index_to_id: {                      # see §5.3
    [index]: tool_call_id,
  },
}
```

### 5.2 On `content_block_start(index, tool_call_id, name)`

Initialize `buffers[tool_call_id]` and record `index_to_id[index] = tool_call_id`.

### 5.3 On `content_block_delta(index, args_delta)`

If the delta carries only `index` (provider may omit `tool_call_id` on subsequent deltas), resolve the target buffer via `index_to_id[index]`. Append `args_delta` to `buffers[tool_call_id].args_fragments`.

### 5.4 On `tool_block_complete(tool_call_id, arguments)`

1. Record `last_sealed_id = tool_call_id`.
2. Dispatch the tool: spawn an async task to execute `tool(arguments)` and buffer the result.
3. Delete `buffers[tool_call_id]` (arguments have been delivered authoritatively by the event).
4. Emit observability span with `seal_latency_ms`.

### 5.5 On `message_stop`

Drain all pending results from the executor pool. Emit results in matching order on the next turn.

### 5.6 Backwards-compatible fallback

If no `tool_block_complete` events arrive, fall back to detecting seals via a new `tool_call_id` on the next delta (current heuristic). `eager-tools-core` uses this fallback automatically when STDP negotiation fails.

---

## 6. Backwards Compatibility

- **Consumers without STDP awareness**: unaffected. New events are silently ignored by standard SSE parsers.
- **Providers without STDP**: unaffected. Request header is advisory; absence of response events means "not supported, fall back to heuristic detection."
- **Intermediate gateways (proxies, load balancers)**: MUST forward `tool_block_complete` events verbatim. Gateways that buffer or reorder SSE events break STDP and classic streaming alike; STDP imposes no new requirements on gateways.

---

## 7. Security Considerations

### 7.1 Tool-call injection via stream manipulation

An attacker with the ability to forge SSE events into a conversation could potentially inject fake `tool_block_complete` events to trigger unintended tool dispatches. This risk is identical to the existing risk of forging `tool_use` blocks and is mitigated by:

- TLS on the provider connection (already required).
- Runtime-side validation that `tool_call_id` on a `tool_block_complete` event matches a `tool_call_id` seen in a prior `content_block_start`.
- Idempotency gating: destructive tools MUST be marked non-idempotent and bypass the eager path.

### 7.2 Tool retraction window

Because eagerly-dispatched tools begin executing before `message_stop`, a model that retracts the call cannot un-execute it. Runtimes MUST NOT eagerly dispatch non-idempotent tools (§4.5). Providers MAY in the future emit `tool_block_retracted` to give runtimes a best-effort cancellation opportunity, but runtimes MUST NOT depend on such an event to maintain safety.

### 7.3 Timing side channels

`sealed_at_ms` leaks provider-side timing information. Providers concerned about fingerprinting or side-channel leakage MAY omit this field.

---

## 8. Provider Landscape (2026)

| Provider | Current tool streaming | STDP mapping |
|----------|------------------------|--------------|
| **Anthropic Claude** | `content_block_start` / `_delta` / `_stop` with typed blocks | `content_block_stop` for a `tool_use` block already carries the needed signal. STDP adoption is mostly a matter of surfacing `sealed_at_ms` and formalizing arguments payload. |
| **OpenAI GPT** | `tool_calls[i]` deltas on `chat.completions.chunk`, closed at `finish_reason: "tool_calls"` | No per-block seal today — runtimes infer completion heuristically. STDP adds the signal that is currently missing. |
| **Google Gemini** | `candidates[].content.parts[]` with `functionCall` parts streamed as full parts | Gemini already emits complete function calls atomically; STDP maps to the existing part boundary. |
| **Amazon Bedrock (Converse API)** | `contentBlockStart` / `contentBlockDelta` / `contentBlockStop` | Same surface as Anthropic's native API; mapping is mechanical. |
| **Mistral** | `tool_calls` array similar to OpenAI | Same heuristic gap as OpenAI; STDP recommended. |
| **Local inference (vLLM, TGI, Ollama)** | Varies by runtime | RECOMMENDED to implement STDP natively when streaming tool calls; reference implementation available in `eager-tools-core`. |

---

## 9. Reference Implementation

[`eager-tools`](https://github.com/cloudthinker-ai/eager-tools) provides:

- **`eager-tools-core`** — provider-agnostic `SealDetector` and `ExecutorPool` implementing the §5 state machine. Passes a golden-trace conformance suite.
- **`eager-tools-anthropic`** — adapter that maps Anthropic SSE events to `SealEvent`s, with automatic fallback to heuristic detection when STDP is absent.
- **`eager-tools-openai`** — adapter for OpenAI chat.completions streaming, same fallback behavior.

The conformance suite (`packages/eager-tools-core/tests/test_seal_detector.py`) documents the expected behavior of a correct implementation and is REQUIRED reading for any new adapter.

---

## 10. Open Questions

1. **Capability discovery.** Should providers expose STDP support on `/v1/models`, via a request header echo, or purely via event-presence? v0.1 uses event-presence; v0.2 may add explicit discovery if demand materializes.
2. **Retraction event.** Should `tool_block_retracted` ship in v0.1 or wait for field data? Current proposal: wait. Retraction is rare in production traces (<0.1%) and runtime-side idempotency gating covers the safety gap.
3. **Streaming result protocol.** This RFC covers the request→tool direction only. A future RFC (STDP-Results) may cover streaming tool results back to the model for extremely long-running tools.
4. **Multi-agent coordination.** When one agent's tool call triggers a subagent that itself streams, how should nested STDP events be labeled? Current view: each subagent stream is its own STDP scope; parent runtime is responsible for correlation via conversation IDs.

---

## 11. Non-Goals

- STDP does not attempt to standardize the shape of `tool_use` blocks, tool-result blocks, or the broader Messages API. Each provider retains full sovereignty over its native surface.
- STDP does not mandate eager dispatch. It exposes the *signal*; runtimes choose whether and when to dispatch eagerly based on tool idempotency and workload characteristics.
- STDP is not an agent protocol. It is a streaming-transport protocol — the thinnest possible layer enabling eager dispatch.

---

## 12. Acknowledgements

This protocol was extracted from production observations at CloudThinker, where eager tool calling delivers 50% median and up to 21× worst-case speedups on real agent workloads. The `tool_call_id` seal mechanism it formalizes is not novel — CPUs have pipelined instruction decode since the 1980s. What is novel is recognizing that agent runtimes are CPUs, and per-block streaming is the register-ready signal.

---

## 13. Appendix — Change Log

- **v0.1 (2026-04-15)**: Initial draft. Minimal event set, opt-in negotiation, reference implementation linked.

## 14. Appendix — Migration Notes

### 14.1 For provider maintainers adopting STDP

1. Identify the internal point at which the model's tool-call arguments buffer stops receiving tokens for a given block.
2. Emit `tool_block_complete` with the full parsed arguments at that point.
3. Expose the `X-Streaming-Tool-Dispatch: v1` header; emit the new event only when set.
4. Run the `eager-tools-core` golden-trace conformance suite against your endpoint.

### 14.2 For runtime authors adopting STDP

1. Set the `X-Streaming-Tool-Dispatch: v1` header on streaming requests.
2. Listen for `tool_block_complete` events in the SSE stream.
3. Fall back to heuristic detection (new `tool_call_id` on the next delta seals the previous) if the event is absent.
4. Route non-idempotent tools to the classic (post–`message_stop`) path.
5. Instrument `seal_latency_ms` for observability.

---

**Feedback and co-signers welcome.** This is a draft. If you maintain an LLM provider, an agent framework, or a significant production runtime, please engage in GitHub Discussions — we'd rather negotiate the shape now than have three incompatible implementations ship in parallel.
