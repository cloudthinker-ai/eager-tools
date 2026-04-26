"""EagerMiddleware — `awrap_model_call` wrapper for `langchain.agents.create_agent`.

Owns the `model.astream(...)` loop, feeds chunks through `SealDetector`, and
dispatches each idempotent tool the moment its JSON block seals. Returns a
`ModelResponse(result=[AIMessage, ToolMessage₁, …])` so the agent commits the
assistant message + eagerly-resolved tool results in a single step via the
`add_messages` reducer.

Non-idempotent tool calls are NOT pre-resolved here — they fall through to the
agent's normal tool step (`ToolNode`) which the framework runs after this
middleware returns the assistant message.

See `~/.claude-duc/plans/plan-langgraph-adapter.md` for the design rationale.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from eager_tools import (
    NOOP_OBSERVABILITY,
    ExecutorPool,
    NonIdempotentToolError,
    ObservabilityHook,
    SealDetector,
    Tool,
    ToolCall,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ToolMessage,
    message_chunk_to_message,
)

from .chunks import normalize_chunk


class EagerMiddleware(AgentMiddleware[Any, Any]):
    """`AgentMiddleware` that overlaps tool execution with model streaming.

    Register on a `create_agent`:

        from langchain.agents import create_agent
        from eager_tools_langgraph import eager_middleware

        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-5"),
            tools=[my_tools],
            middleware=[eager_middleware({t.name: t for t in my_tools})],
        )

    The middleware:
    1. Owns the `model.astream(...)` loop on every model step.
    2. Routes streamed `tool_call_chunks` through `SealDetector`.
    3. Dispatches each idempotent tool the moment it seals.
    4. After stream end, drains in-flight results and returns the assistant
       `AIMessage` plus one `ToolMessage` per eagerly-resolved tool_call_id —
       all in `ModelResponse.result`, committed atomically via `add_messages`.
    5. Non-idempotent tool_call_ids are skipped here. The agent's normal tool
       step picks them up.
    """

    def __init__(
        self,
        tools: dict[str, Tool],
        *,
        observability: ObservabilityHook = NOOP_OBSERVABILITY,
        max_concurrent: int = 32,
        conversation_id: str | None = None,
    ) -> None:
        super().__init__()
        self._tools = tools
        self._observability = observability
        self._max_concurrent = max_concurrent
        self._conversation_id = conversation_id

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        # Why this exists despite being a no-op: `langchain.agents.factory`
        # only wires the model-destination edge into the post-model branch
        # when at least one middleware overrides `after_model` (or there's a
        # `response_format`). Without this, returning a `ModelResponse` with
        # eager-resolved ToolMessages crashes with `KeyError: 'model'` because
        # the routing function falls through to case 6 (all tool_calls already
        # resolved) and returns `model_destination`, which isn't in the
        # branch's `ends`. Overriding `after_model` flips `loop_exit_node` to
        # a separate node, which adds `loop_entry_node` to the destinations.
        # See `langchain/agents/factory.py:1514-1516`.
        del state, runtime
        return None

    async def awrap_model_call(  # type: ignore[override]
        self,
        request: ModelRequest[Any],
        handler: Any,
    ) -> ModelResponse[Any]:
        del handler  # We own the model call entirely; no fallthrough needed.

        detector = SealDetector(conversation_id=self._conversation_id)
        pool = ExecutorPool(
            self._tools,
            observability=self._observability,
            max_concurrent=self._max_concurrent,
        )
        eager_ids: set[str] = set()
        merged: AIMessageChunk | None = None

        try:
            # NOTE: do NOT re-pass `request.tools` here. `create_agent` has
            # already bound them on `request.model` before invoking middleware;
            # passing `tools=` again either duplicates the binding (rejected by
            # some providers) or overrides `tool_choice` set by the model.
            stream = request.model.astream(
                request.messages,
                **(request.model_settings or {}),
            )
            try:
                async for chunk in stream:
                    if not isinstance(chunk, AIMessageChunk):
                        continue
                    merged = chunk if merged is None else merged + chunk
                    for nc in normalize_chunk(chunk):
                        seal = detector.observe(**nc.as_kwargs())
                        if seal is not None and seal.tool_call is not None:
                            await self._handle_seal(pool, seal, eager_ids)
            finally:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    with contextlib.suppress(Exception):
                        await aclose()

            final = detector.finalize()
            if final is not None and final.tool_call is not None:
                await self._handle_seal(pool, final, eager_ids)

            if merged is None:
                return ModelResponse(result=[AIMessage(content="")])

            ai_msg = message_chunk_to_message(merged)
            assert isinstance(ai_msg, AIMessage)

            tool_msgs = await self._drain_tool_results(pool, ai_msg.tool_calls, eager_ids)
            return ModelResponse(result=[ai_msg, *tool_msgs])
        except (asyncio.CancelledError, GeneratorExit):
            await pool.cancel_all()
            raise
        finally:
            await pool.close()

    async def _handle_seal(
        self,
        pool: ExecutorPool,
        seal: Any,
        eager_ids: set[str],
    ) -> None:
        """Route a SealEvent: parse_error → record as tool error; otherwise dispatch.

        Tracks `tool_call_id` in `eager_ids` for both branches so the
        middleware emits a ToolMessage for the bad call too — leaving it for
        the agent's normal tool step would cause a re-execution of an
        already-malformed call.
        """
        assert seal.tool_call is not None
        # Sync call — `ObservabilityHook.on_seal` is `def`, not `async def`.
        # Suppress observer errors to match `ExecutorPool._safe_hook` semantics:
        # a broken hook must never abort the stream.
        with contextlib.suppress(Exception):
            self._observability.on_seal(seal)
        if seal.parse_error is not None:
            await pool.record_error(seal.tool_call, seal.parse_error)
            eager_ids.add(seal.tool_call.tool_call_id)
            return
        await self._dispatch_if_idempotent(pool, seal.tool_call, eager_ids)

    async def _dispatch_if_idempotent(
        self,
        pool: ExecutorPool,
        call: ToolCall,
        eager_ids: set[str],
    ) -> None:
        """Try eager dispatch. Idempotent tool → fire. Non-idempotent → skip silently."""
        try:
            await pool.dispatch(call)
            eager_ids.add(call.tool_call_id)
        except NonIdempotentToolError:
            pass

    async def _drain_tool_results(
        self,
        pool: ExecutorPool,
        ai_tool_calls: list[dict[str, Any]],
        eager_ids: set[str],
    ) -> list[ToolMessage]:
        """Wait for every dispatched tool, emit ToolMessages keyed by tool_call_id.

        Non-idempotent tool_call_ids in `ai_tool_calls` are NOT pre-resolved —
        the agent's normal tool step handles them. We only emit ToolMessages
        for ids in `eager_ids`.

        Pool results may arrive in any order; we collect into a dict and emit in
        the order tool_calls appear on the AIMessage so downstream reducers see
        a stable sequence.
        """
        if not eager_ids:
            return []

        await pool.close()  # Signal: no more dispatches; let results() drain.
        results: dict[str, ToolMessage] = {}
        async for call, payload in pool.results():
            results[call.tool_call_id] = _build_tool_message(call, payload)

        ordered: list[ToolMessage] = []
        seen: set[str] = set()
        for tc in ai_tool_calls:
            tc_id = tc.get("id")
            if tc_id in eager_ids and tc_id in results:
                ordered.append(results[tc_id])
                seen.add(tc_id)
        # Parse-error seals are absent from `ai_msg.tool_calls` (LangChain
        # routes malformed entries to `invalid_tool_calls` instead). Append
        # them so the model still sees the error and can recover.
        for tc_id, msg in results.items():
            if tc_id not in seen:
                ordered.append(msg)
        return ordered


def _build_tool_message(call: ToolCall, payload: Any) -> ToolMessage:
    """Wrap a pool result into a `ToolMessage` keyed by `call.tool_call_id`.

    Exceptions become error-status ToolMessages so the agent step does not
    crash — the model can see the error on the next turn and recover.
    """
    if isinstance(payload, BaseException):
        return ToolMessage(
            content=f"{type(payload).__name__}: {payload}",
            tool_call_id=call.tool_call_id,
            name=call.name,
            status="error",
        )
    if isinstance(payload, str):
        content: Any = payload
    else:
        try:
            content = json.dumps(payload, default=str)
        except (TypeError, ValueError):
            content = str(payload)
    return ToolMessage(
        content=content,
        tool_call_id=call.tool_call_id,
        name=call.name,
    )


def eager_middleware(
    tools: dict[str, Tool],
    *,
    observability: ObservabilityHook = NOOP_OBSERVABILITY,
    max_concurrent: int = 32,
    conversation_id: str | None = None,
) -> EagerMiddleware:
    """Convenience factory mirroring `AnthropicEagerStream(...)` ergonomics."""
    return EagerMiddleware(
        tools,
        observability=observability,
        max_concurrent=max_concurrent,
        conversation_id=conversation_id,
    )


__all__ = ["EagerMiddleware", "eager_middleware"]
