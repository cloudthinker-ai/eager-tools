"""ExecutorPool contract tests.

Every test uses in-process fake tools — no network, no SDK.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from eager_tools import (
    EagerDispatchDeniedError,
    ExecutorPool,
    GateDeniedError,
    NonIdempotentToolError,
    SealEvent,
    ToolCall,
)


class FakeTool:
    def __init__(
        self,
        name: str,
        *,
        idempotent: bool = True,
        delay: float = 0,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.idempotent = idempotent
        self._delay = delay
        self._error = error

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return {"ok": True, **arguments}


def _call(name: str, **kwargs: Any) -> ToolCall:
    return ToolCall(tool_call_id=f"call-{name}", name=name, arguments=kwargs)


async def test_dispatch_idempotent_tool_runs_to_completion() -> None:
    tool = FakeTool("read_file")
    pool = ExecutorPool({"read_file": tool})
    await pool.dispatch(_call("read_file", path="/etc"))
    results: list[tuple[ToolCall, Any]] = []
    await pool.close()
    async for item in pool.results():
        results.append(item)
    assert len(results) == 1
    call, result = results[0]
    assert call.name == "read_file"
    assert result == {"ok": True, "path": "/etc"}


async def test_dispatch_non_idempotent_raises() -> None:
    tool = FakeTool("send_email", idempotent=False)
    pool = ExecutorPool({"send_email": tool})
    with pytest.raises(NonIdempotentToolError):
        await pool.dispatch(_call("send_email"))
    await pool.close()


async def test_dispatch_unknown_tool_surfaces_keyerror_in_results() -> None:
    pool = ExecutorPool({})
    await pool.dispatch(_call("missing_tool"))
    await pool.close()
    results = [item async for item in pool.results()]
    assert len(results) == 1
    call, exc = results[0]
    assert call.name == "missing_tool"
    assert isinstance(exc, KeyError)


async def test_errors_isolated_between_tools() -> None:
    tools = {
        "good_1": FakeTool("good_1"),
        "bad": FakeTool("bad", error=RuntimeError("boom")),
        "good_2": FakeTool("good_2"),
    }
    pool = ExecutorPool(tools)
    await pool.dispatch(_call("good_1"))
    await pool.dispatch(_call("bad"))
    await pool.dispatch(_call("good_2"))
    await pool.close()

    results = {r[0].name: r[1] async for r in pool.results()}
    assert isinstance(results["bad"], RuntimeError)
    assert results["good_1"] == {"ok": True}
    assert results["good_2"] == {"ok": True}


async def test_cancel_all_propagates_cancellation() -> None:
    tool = FakeTool("slow", delay=10.0)
    pool = ExecutorPool({"slow": tool})
    await pool.dispatch(_call("slow"))
    assert pool.in_flight == 1
    await pool.cancel_all()
    assert pool.in_flight == 0


async def test_close_ends_results_iteration() -> None:
    tool = FakeTool("fast")
    pool = ExecutorPool({"fast": tool})
    await pool.dispatch(_call("fast"))
    await pool.close()
    results = [item async for item in pool.results()]
    assert len(results) == 1


async def test_max_concurrent_respected() -> None:
    peak = 0
    current = 0
    lock = asyncio.Lock()

    class InstrumentedTool:
        name = "slow"
        idempotent = True

        async def __call__(self, arguments: dict[str, Any]) -> Any:
            nonlocal peak, current
            async with lock:
                current += 1
                if current > peak:
                    peak = current
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1
            return {"ok": True}

    tools: dict[str, Any] = {"slow": InstrumentedTool()}
    pool = ExecutorPool(tools, max_concurrent=2)

    for i in range(6):
        await pool.dispatch(ToolCall(tool_call_id=f"call-{i}", name="slow", arguments={}))
    await pool.close()
    _ = [item async for item in pool.results()]
    assert peak <= 2


async def test_in_flight_property_tracks_live_tasks() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingTool:
        name = "block"
        idempotent = True

        async def __call__(self, arguments: dict[str, Any]) -> Any:
            started.set()
            await release.wait()
            return {"ok": True}

    pool = ExecutorPool({"block": BlockingTool()})
    assert pool.in_flight == 0
    await pool.dispatch(_call("block"))
    await started.wait()
    assert pool.in_flight == 1
    release.set()
    await pool.close()
    _ = [item async for item in pool.results()]
    assert pool.in_flight == 0


async def test_gate_allows_dispatch() -> None:
    """A tool whose `gate` returns True dispatches normally."""

    class GatedTool:
        name = "read_file"
        idempotent = True

        async def gate(self, call: ToolCall) -> bool:
            del call
            return True

        async def __call__(self, arguments: dict[str, Any]) -> Any:
            return {"path": arguments.get("path"), "ran": True}

    pool = ExecutorPool({"read_file": GatedTool()})
    await pool.dispatch(_call("read_file", path="/var/log/app.log"))
    await pool.close()
    results = [item async for item in pool.results()]
    assert len(results) == 1
    _, payload = results[0]
    assert payload == {"path": "/var/log/app.log", "ran": True}


async def test_gate_denies_raises_gate_denied_error() -> None:
    """Gate returning False raises `GateDeniedError`. `EagerDispatchDeniedError`
    catches it; `NonIdempotentToolError` does NOT (sibling, not subclass).
    Pool does NOT push to results on this path.
    """

    class DenyingTool:
        name = "read_file"
        idempotent = True

        async def gate(self, call: ToolCall) -> bool:
            del call
            return False

        async def __call__(self, arguments: dict[str, Any]) -> Any:
            del arguments
            return {"unreachable": True}

    pool = ExecutorPool({"read_file": DenyingTool()})
    with pytest.raises(GateDeniedError) as info:
        await pool.dispatch(_call("read_file", path="/etc/shadow"))
    # Hierarchy contract: catchable as the shared base.
    assert isinstance(info.value, EagerDispatchDeniedError)
    # Sibling, not subclass — different remediation paths.
    assert not isinstance(info.value, NonIdempotentToolError)

    await pool.close()
    results = [item async for item in pool.results()]
    assert results == []


async def test_gate_exception_chains_via_cause() -> None:
    """If gate raises, the original exception is chained on `__cause__`."""

    class RaisingGateTool:
        name = "read_file"
        idempotent = True

        async def gate(self, call: ToolCall) -> bool:
            del call
            raise ValueError("policy denied")

        async def __call__(self, arguments: dict[str, Any]) -> Any:
            del arguments
            return None

    pool = ExecutorPool({"read_file": RaisingGateTool()})
    with pytest.raises(GateDeniedError) as info:
        await pool.dispatch(_call("read_file"))
    assert isinstance(info.value.__cause__, ValueError)
    assert str(info.value.__cause__) == "policy denied"

    await pool.close()
    results = [item async for item in pool.results()]
    assert results == []


async def test_gate_cancellation_propagates() -> None:
    """Cancelling the dispatch task while the gate is awaiting cancels the gate.

    Mirrors the tool execution contract: gates with side effects are
    responsible for their own cleanup via try/finally.
    """
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class HangingGateTool:
        name = "slow"
        idempotent = True

        async def gate(self, call: ToolCall) -> bool:
            del call
            started.set()
            try:
                await asyncio.sleep(60)
                return True
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def __call__(self, arguments: dict[str, Any]) -> Any:
            del arguments
            return None

    pool = ExecutorPool({"slow": HangingGateTool()})
    task = asyncio.create_task(pool.dispatch(_call("slow")))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


async def test_non_idempotent_short_circuits_before_gate() -> None:
    """Idempotency check runs first. A non-idempotent tool with a gate must
    raise `NonIdempotentToolError` and the gate must NOT be invoked. Locks
    in the per-tool blanket-deny invariant so a future refactor that swaps
    the order (gate-first) can't silently let gates with side effects fire
    for non-idempotent tools.
    """

    class UnsafeGatedTool:
        def __init__(self) -> None:
            self.name = "send_email"
            self.idempotent = False
            self.gate_called = False

        async def gate(self, call: ToolCall) -> bool:
            del call
            self.gate_called = True
            return True

        async def __call__(self, arguments: dict[str, Any]) -> Any:
            del arguments
            return "sent"

    tool = UnsafeGatedTool()
    pool = ExecutorPool({"send_email": tool})
    with pytest.raises(NonIdempotentToolError):
        await pool.dispatch(_call("send_email"))
    assert tool.gate_called is False
    await pool.close()


async def test_gate_fn_alias_reexported() -> None:
    """Smoke: `GateFn` is importable from the public package and matches the
    documented `Callable[[ToolCall], Awaitable[bool]]` signature shape.
    """
    from eager_tools import GateFn

    assert GateFn is not None
    # Concrete async callable taking ToolCall → bool satisfies the alias at
    # runtime (typing-only check; this just pins the re-export).

    async def _gate(call: ToolCall) -> bool:
        del call
        return True

    fn: GateFn = _gate  # noqa: F841 — pin assignment compatibility


async def test_no_gate_attribute_dispatches_normally() -> None:
    """Tools without a `gate` attribute are unaffected — duck-typed lookup."""
    tool = FakeTool("read_file")
    assert not hasattr(tool, "gate")
    pool = ExecutorPool({"read_file": tool})
    await pool.dispatch(_call("read_file", path="/tmp"))
    await pool.close()
    results = [item async for item in pool.results()]
    assert len(results) == 1
    _, payload = results[0]
    assert payload == {"ok": True, "path": "/tmp"}


async def test_observer_exception_does_not_break_dispatch() -> None:
    class BrokenObserver:
        def on_seal(self, event: SealEvent) -> None:
            raise RuntimeError("observer broken")

        def on_dispatch_start(self, call: ToolCall) -> None:
            raise RuntimeError("observer broken")

        def on_dispatch_end(self, call: ToolCall, error: Exception | None) -> None:
            raise RuntimeError("observer broken")

    tool = FakeTool("read_file")
    pool = ExecutorPool({"read_file": tool}, observability=BrokenObserver())
    await pool.dispatch(_call("read_file"))
    await pool.close()
    results = [item async for item in pool.results()]
    assert len(results) == 1
    _, result = results[0]
    assert result == {"ok": True}
