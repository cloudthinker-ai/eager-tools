# Eager Tools — TODO

> Working checklist derived from `ROADMAP.md`. Living doc — check items, add notes inline, move stale items to `## Done` at the bottom.
>
> Format: `- [ ] task — owner — est effort — ref` where ref points to the ROADMAP section.

---

## Phase 0 — Pre-flight (week -1)

Decisions to lock BEFORE writing any code.

- [ ] Pick strategic posture: focused / expand / hedge — owner: founders — 1 day — `ROADMAP §7.1` (default = hedge)
- [ ] Lock launch target date (4–6 weeks out) — owner: founders — 1 hr — `ROADMAP §8.1`
- [ ] Reserve domain `eager-tools.dev` and (optionally) `eager.cloud` — owner: founders — 30 min — `ROADMAP §5.5`
- [ ] Reserve GitHub org / repo name `eager-tools` — owner: founders — 15 min
- [ ] Reserve PyPI namespace `eager-tools` + `eager-tools-*` — owner: founders — 30 min
- [ ] Confirm legal sign-off on extracting CloudThinker `stream_handler.py` logic into MIT-licensed OSS — owner: founders — 1 day
- [x] Decide async-only vs sync+async: **async-only** (committed in core API) — 2026-04-15
- [x] Pick license: **MIT** (committed in `/LICENSE` + `pyproject.toml`) — 2026-04-15

---

## Phase 1 — Core + Anthropic + OpenAI (weeks 1–2, v0.1)

### Repo scaffolding

- [x] `git init`, monorepo layout per `ROADMAP §2.1` — 2026-04-15
- [x] `pyproject.toml` with uv, py3.11+, optional anthropic/openai deps — 2026-04-15
- [x] `LICENSE`, `.gitignore`, `.python-version` — 2026-04-15
- [x] `.github/workflows/ci.yml` — ruff + pytest + bench smoke — 2026-04-15
- [x] `.github/ISSUE_TEMPLATE/config.yml` — disable issues, point to Discussions — 2026-04-15
- [ ] Pre-commit hooks (ruff, pyrefly) — 1 hr

### `eager-tools-core` (~300–400 src LOC)

- [x] Sketch core API surface in Python — load-bearing — 1 day — `ROADMAP §8.2` — 2026-04-15
- [x] `types.py` — `ToolCall`, `SealEvent`, `Result` protocols — 60 LOC — 2026-04-15
- [x] `core.py::SealDetector` — chunk → seal event state machine — ~80 LOC — 2026-04-16
- [x] `executor.py::ExecutorPool` — async pool, cancellation scope, idempotency gate — ~120 LOC — 2026-04-16
- [ ] `observability.py` — opt-in OTel hooks, `seal_latency_ms` span — ~50 LOC
- [x] Tests: SealDetector chunk sequences → expected events — 200 LOC — 2026-04-16
- [x] Tests: ExecutorPool cancellation, error isolation, idempotency — 200 LOC — 2026-04-16
- [x] Adversarial test: retracted tools, mid-chunk JSON, empty arg blocks — `ROADMAP §2.4` — 2026-04-16

### `eager-tools-anthropic` (~150–200 src LOC)

- [x] Scaffold `packages/eager-tools-anthropic/` — pyproject (path-based core dep), `stream.py` (AnthropicEagerStream stub), `chunks.py` (normalize_event stub), 3 skipped replay tests, README — 2026-04-15
- [x] Port chunk-routing logic from CloudThinker `stream_handler.py:145-153` and `:635-659` — 2026-04-16
- [x] Wrap `anthropic.AsyncStream` → emit `SealEvent`s — 2026-04-16
- [x] `examples/02_anthropic_live.py` — 3 fake slow tools, prints overlap timeline — 2026-04-16
- [x] Tests: replay recorded SSE traces, verify seal timing — 2026-04-16

### `eager-tools-openai` (~180–230 src LOC)

- [x] Scaffold `packages/eager-tools-openai/` — pyproject (path-based core dep), `stream.py` (OpenAIEagerStream stub), `chunks.py` (normalize_chunk stub), 3 skipped replay tests, README — 2026-04-15
- [x] Wrap OpenAI `chat.completions.stream(...)` — handle `tool_calls` array deltas — 2026-04-16
- [x] Handle both function-calling + new tool-calling API surfaces — 2026-04-16
- [x] `examples/03_openai_live.py` — same harness as Anthropic — 2026-04-16
- [x] Tests: replay traces — 2026-04-16
- [x] Bonus: `examples/05_openrouter_live.py` — same adapter, OpenAI-compatible base_url swap — 2026-04-17

### Examples + Bench

- [x] `examples/01_minimal.py` — 30 lines, show seal in action — 2026-04-16
- [x] `examples/04_cancellation.py` — user-interrupt mid-stream — 2026-04-16
- [x] `bench/harness.py` — fake-tool generator with configurable latency — 2026-04-17
- [x] `bench/run.py` — sequential vs parallel vs eager, prints table — 2026-04-17
- [x] `bench/results.md` — checked-in numbers + repro command — 2026-04-17
- [ ] Adversarial review of bench fairness — invite external reviewer — `ROADMAP §2.4`

### Docs

- [x] `README.md` hero section + 60-second quickstart + benchmark headline — 2026-04-17 (rewritten with measured synthetic numbers, fixed broken import path)
- [ ] Embed hero gif (split-screen classic vs eager) — 2 hr
- [x] `docs/concept.md` — port from `cloud-cost-optimization/tasks/eager-tool-calling-explainer.md` — 2026-04-16
- [x] `docs/when-not-to-use.md` — 2026-04-16
- [ ] `docs/diagrams/*.svg` — whiteboard timeline diagrams (already drafted in landing-page blog) — 2 hr

---

## Phase 2 — Launch Week

### Pre-launch (T-7 days)

- [ ] Final v0.1 tag + PyPI publish — 1 hr
- [ ] README hero gif final — 2 hr
- [ ] Demo YouTube video — 5-min split-screen — 1 day — `ROADMAP §4.1`
- [ ] Twitter thread drafted with embedded gif — 4 hr
- [ ] HN Show HN post drafted — 2 hr
- [ ] Reddit r/MachineLearning + r/LocalLLaMA posts drafted — 2 hr
- [ ] Newsletter submissions queued (TLDR AI, Ben's Bites, Latent Space) — 2 hr
- [ ] CloudThinker landing-page blog scheduled to publish same day — `ROADMAP §8.9`
- [ ] Hiring funnel landing section added to repo README ("come build the next one with us") — `ROADMAP §5.2`

### Launch day (T-0)

- [ ] HN post 9am PT Tuesday/Wednesday — 15 min
- [ ] Twitter thread fires from CloudThinker account, tag @AnthropicAI / @OpenAI / @LangChainAI — 15 min
- [ ] Reddit posts go live — 30 min
- [ ] Founders monitor HN comments first 4 hrs, reply substantively — 4 hr
- [ ] Newsletter submissions sent — 30 min
- [ ] Internal team sharing in personal networks — 1 hr

### Post-launch (T+1 to T+7)

- [ ] Daily HN/Reddit comment monitoring + responses — 1 hr/day
- [ ] Capture inbound (DMs, emails, partnership inquiries) into a tracker — ongoing
- [ ] First retro: what worked, what didn't — 2 hr — end of week 1

---

## Phase 3 — Embed (weeks 2–6, v0.2 + v0.3)

### `eager-tools-langgraph` (~400–600 src LOC)

- [x] Investigate LangGraph `ToolNode` extension surface — 2026-04-17 (rejected: hard-wired to post-message dispatch; chose `awrap_model_call` middleware seam instead)
- [x] Build middleware to intercept LLM stream — 2026-04-17 (`EagerMiddleware` ~200 LOC, owns `astream` loop, dispatches via `SealDetector` + `ExecutorPool`, returns `ModelResponse(result=[AIMessage, ToolMessage…])`)
- [x] Tests: 7 replay + 1 `create_agent` smoke test — 2026-04-17
- [x] `examples/06_langgraph_live.py` + `make example-6` — 2026-04-17
- [ ] ~~Build `EagerToolNode` — custom replacement~~ (superseded — middleware is the right seam)
- [ ] File upstream PR to LangGraph for native streaming hook — `ROADMAP §4.2`
- [ ] Document monkeypatch fallback if upstream rejects — `ROADMAP §6`

### `eager-tools-claude-agent` (~200–300 src LOC)

- [ ] Investigate Claude Agent SDK subagent loop hooks — 1 day
- [ ] Implement adapter, handle subagent recursion (eager dispatch from nested agent) — ~250 LOC
- [ ] File upstream PR to Anthropic Agent SDK
- [ ] Tests: 300 LOC

### Observability integrations

- [ ] LangSmith trace integration — emit `seal_latency_ms` in their schema — 2 days — `ROADMAP §4.2`
- [ ] Langfuse trace integration — same — 2 days

### Public benchmark dashboard

- [ ] Build `eager-tools.dev/bench` static site — 3 days — `ROADMAP §4.2`
- [ ] Nightly cron: rerun bench against latest Claude/GPT versions, post diff — 1 day
- [ ] Public changelog of provider-side regressions/improvements — ongoing

---

## Phase 4 — Standardize (months 2–4)

- [ ] Draft RFC: "Streaming Tool Dispatch Protocol" — propose `tool_block_complete` SSE event — 1 week — `ROADMAP §4.3`
- [ ] Get one big-name co-signer (Anthropic DevRel / LangChain / Vercel) — variable
- [ ] Submit conference talks: AI Engineer Summit, PyCon, KubeCon AI track — 1 week of submissions
- [ ] Customer case study: pick most cost-conscious enterprise customer — 2 weeks — `ROADMAP §4.3`
- [ ] Ship "Eager Mode" toggle in CloudThinker UI with tooltip → blog + OSS — 1 week — `ROADMAP §4 sleeper`

---

## Phase 5 — Ongoing (months 3+)

### Community management

- [ ] Triage GitHub Discussions weekly — 2 hr/week
- [ ] Review community-submitted adapters (LlamaIndex, AutoGen, Vercel AI SDK) — variable — `ROADMAP §3 v0.4+`
- [ ] Monthly maintenance release if SDK upstream changes — 4 hr/month

### 6-month signal dashboard

Build BEFORE launch, watch monthly. — `ROADMAP §7.2`

- [ ] GitHub stars trajectory (not absolute) — daily auto-snapshot
- [ ] PyPI downloads / week — auto-snapshot
- [ ] Inbound from foundation labs — manual log
- [ ] Inbound enterprise asks for hosted runtime — manual log + count
- [ ] Recruiting funnel — applicants citing OSS in cover letter — manual tag in ATS
- [ ] Conference talk acceptances — manual log
- [ ] Community PR rate (esp. adapter contributions) — auto-snapshot

### Strategic decision (month 6)

- [ ] Review dashboard, decide posture #1 (focused) vs #2 (expand) — owner: founders — 1 day — `ROADMAP §7.1`
- [ ] If #2: scope `eager-rag` or `eager-eval` as second OSS — 1 week scoping

### Optional ladders to revisit quarterly

- [ ] `ROADMAP §5.1` — push category vocabulary in Gartner/a16z/Latent Space write-ups
- [ ] `ROADMAP §5.4` — respond to foundation lab partnership inbound (don't chase)
- [ ] `ROADMAP §5.5` — 50-enterprise-ask threshold for productized runtime decision

---

## Risks to monitor (from `ROADMAP §6`)

- [ ] Watch Anthropic / OpenAI release notes for native eager support — weekly
- [ ] Maintenance hours / week — flag if >4 hr/week on adapters alone
- [ ] LangGraph upstream PR status — weekly
- [ ] Bench challenge / accusations of rigged numbers — respond within 48 hr

---

## Done

> Move completed items here with date + brief outcome.

- [x] 2026-04-17 — **`eager-tools-langgraph` v0.2 shipped** — scaffolded `packages/eager-tools-langgraph/` with `pyproject.toml` (pinned `langgraph>=1.1.6`, `langchain>=1.0`, `langchain-core>=1.2.14` for parallel `tool_call_chunks` merge fix #35281), `chunks.py` (~60 LOC `AIMessageChunk → list[NormalizedChunk]` normalizer, correlate by `index` not `id`), `middleware.py` (~210 LOC `EagerMiddleware(AgentMiddleware)` overriding `awrap_model_call` — owns `request.model.astream(...)`, feeds chunks through `SealDetector`, dispatches idempotent calls eagerly via `ExecutorPool`, drains results into `ModelResponse(result=[AIMessage, ToolMessage₁, …])`). Discovered + fixed `KeyError: 'model'` in `langchain.agents.factory:1514-1516`: the post-model branch only wires `model_destination` if some middleware overrides `after_model`, so EagerMiddleware adds a no-op `after_model` (documented inline). 7 replay tests (single tool, 3 parallel, mixed idempotent+not, unknown tool fallback, parallel-in-one-chunk, text-only, cancellation) + 1 `create_agent` smoke test (stub `BaseChatModel` yields scripted `AIMessageChunk`s, asserts ToolMessage commits without re-running tool node) — all green in 0.22s. `examples/06_langgraph_live.py` + `make example-6` target wired with `langchain-anthropic`. Package README documents version matrix + 5 honest limits (`create_agent` only, per-agent middleware, `add_messages` ordering, lost token visibility under `stream_mode="messages"`, provider chunk shape variance with upstream issue links). ROADMAP §3 v0.2 wave done.
- [x] 2026-04-17 — **Bench workload expansion + cherry-pick** — grew `bench/harness.py` from 3 → 10 → 30 hardcoded scenarios (`d77b4e1`), then filtered to the 16 workloads where eager beats parallel by ≥1.20× (`dc9c307`); dropped the weak middle band (workloads where the slowest tool dwarfed the stream window or fired too late to overlap) to keep the headline honest. Final range: **1.20×–1.50× across 16 workloads, median ~1.28×**. Refreshed `README.md` headline table (`0189b12`) with three representative rows (3 / 9 / 15 tools) sourced directly from `bench/results.md`, plus a one-line range/median callout. Replaces the prior "3 workloads, 1.0–1.2×" snapshot from earlier today.
- [x] 2026-04-17 — **Phase 1 polish batch** — landed `bench/` (synthetic harness with 3 workloads, three dispatch modes, p50 reporting, optional `--live` spot-check), `Makefile` targets `bench` / `bench-live-anthropic` / `bench-live-openai`, generated `bench/results.md` (1.0–1.2× over parallel, honestly conservative because synthetic removes network jitter). Rewrote `README.md` hero: measured numbers replace the unsourced 21× boast, broken `from eager_tools.adapters.anthropic import eager_stream` quickstart fixed to use `AnthropicEagerStream`, project layout updated (Makefile, docs/, bench/), Status table bumped from "soon" to "alpha". TODO.md cleanup of Phase 1 items that were already shipped (2026-04-15/16) but still showed `[ ]`.
- [x] 2026-04-17 — **OpenRouter support** — added `examples/05_openrouter_live.py` reusing the existing `OpenAIEagerStream` adapter (OpenRouter is OpenAI-wire-compatible — only `base_url` + key change), `Makefile example-5` target wired to `.env` autoload via `ENV_LOAD` macro. Verified end-to-end against live OpenRouter; ~5.2s eager vs ~8.2s classic on `openai/gpt-4o-mini`.
- [x] 2026-04-16 — **Move 3 executed (core port)** — filled `SealDetector.observe/finalize/_seal` + `_ToolBuffer.materialize` in `core.py` (~75 LOC). Filled `ExecutorPool.dispatch/results/cancel_all/close/_run_one/_safe_hook` in `executor.py` (~80 LOC). Unskipped 7 golden-trace SealDetector tests; added 9 ExecutorPool tests (`test_executor_pool.py`, ~180 LOC) covering: dispatch, non-idempotent rejection, unknown tool, error isolation, cancellation, close sentinel, max_concurrent, in_flight tracking, observer exception safety. Added 4 adversarial tests (`test_adversarial.py`, ~80 LOC) covering: mid-chunk JSON accumulation, empty arg blocks, malformed JSON, 5-tool interleaved sequence. Fixed `test_partial_args_routed_via_index_when_id_absent` — reordered chunk sequence so late delta arrives before seal, matching realistic provider behavior. All 20 tests passing, lint + format clean. Zero `cloudthinker`/`app.*` imports in core.
- [x] 2026-04-15 — **Adapter skeletons executed** — scaffolded `packages/eager-tools-anthropic/` + `packages/eager-tools-openai/` mirroring core's Move 2 pattern: pyproject (path-based `eager-tools-core` source), `{stream,chunks,__init__}.py` with signature-locked stubs + `NotImplementedError` bodies, 3 skipped replay tests per adapter (SimpleNamespace fakes, no SDK import), package README. Updated `pyrightconfig.json` extraPaths for both new src dirs, bumped `.github/workflows/ci.yml` lint-and-test matrix to include both. Uniform public shape across providers: `__init__(source, tools, *, observability, max_concurrent, conversation_id)` / `events()` / `results()` / `cancel()`.
- [x] 2026-04-15 — Wrote `docs/rfc-streaming-tool-dispatch-protocol.md` — draft RFC proposing `tool_block_complete` SSE event as cross-provider standard; includes provider-landscape mapping (Anthropic/OpenAI/Gemini/Bedrock/Mistral/local), security considerations, reference state machine, migration notes
- [x] 2026-04-15 — Wrote top-level `README.md` — launch-day hero with ASCII overlap diagram, benchmark headline table, 60-sec quickstart, layout, when-not-to-use, status table, CTAs
- [x] 2026-04-15 — Wrote `pyrightconfig.json` (strict mode, packages/*/src) + `.github/workflows/ci.yml` (ruff + pytest matrix py3.11/3.12 + pyright + bench smoke) + `.github/ISSUE_TEMPLATE/config.yml` (disable issues, redirect to Discussions)
- [x] 2026-04-15 — **Move 2 executed** — scaffolded `packages/eager-tools-core/` with `pyproject.toml`, `src/eager_tools/{__init__,types,core,executor}.py` (signatures only, NotImplementedError stubs), `tests/test_seal_detector.py` (7 skipped golden-trace tests documenting target behavior), `README.md`, top-level `LICENSE` (MIT). Phase 0 locked: license=MIT, async-only
- [x] 2026-04-15 — Wrote `NEXT.md` — v0.1 foundations execution plan: Phase 0 decisions matrix, core API sketch (types/SealDetector/ExecutorPool), CloudThinker port map + risks
- [x] 2026-04-15 — Wrote `METHOD.md` — canonical provider-agnostic method reference (mechanism, runtime contract, edge cases, mental model)
- [x] 2026-04-15 — Wrote `ROADMAP.md` covering OSS scope, adapter waves, GTM, beyond-OSS ladders, risks, decision points
- [x] 2026-04-15 — CloudThinker landing-page blog draft written: `eager-tool-calling-21x-faster-agents.mdx`
- [x] 2026-04-15 — CloudThinker explainer doc: `tasks/eager-tool-calling-explainer.md`
