# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
"""Bench runner — sequential vs parallel vs eager dispatch.

Usage:
    python bench/run.py                        # all workloads, 5 runs, synthetic
    python bench/run.py --workload 8           # one workload
    python bench/run.py --runs 3
    python bench/run.py --live anthropic       # spot-check against real Anthropic
    python bench/run.py --live openai          # or OpenAI
"""

from __future__ import annotations

import argparse
import asyncio
import platform
import statistics
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

# Allow `python bench/run.py` from repo root to import the bench package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import (  # noqa: E402
    WORKLOADS,
    ToolSpec,
    Workload,
    build_tools,
    fake_stream,
)
from eager_tools_anthropic import AnthropicEagerStream  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "bench" / "results.md"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


async def run_eager(workload: Workload) -> float:
    tools = build_tools(workload)
    stream = AnthropicEagerStream(fake_stream(workload), tools=tools)  # type: ignore[arg-type]
    t0 = time.perf_counter()
    async for _ev in stream.events():
        pass
    async for _call, _result in stream.results():
        pass
    return (time.perf_counter() - t0) * 1000.0


async def _drain_stream_collect_calls(workload: Workload) -> list[ToolSpec]:
    """Drain the synthetic stream end-to-end (paced), return the tool list to dispatch."""
    async for _ev in fake_stream(workload):
        pass
    return list(workload.tools)


async def run_parallel(workload: Workload) -> float:
    tools = build_tools(workload)
    t0 = time.perf_counter()
    calls = await _drain_stream_collect_calls(workload)
    await asyncio.gather(*(tools[c.name]({}) for c in calls))
    return (time.perf_counter() - t0) * 1000.0


async def run_sequential(workload: Workload) -> float:
    tools = build_tools(workload)
    t0 = time.perf_counter()
    calls = await _drain_stream_collect_calls(workload)
    for c in calls:
        await tools[c.name]({})
    return (time.perf_counter() - t0) * 1000.0


MODES: dict[str, Any] = {
    "sequential": run_sequential,
    "parallel": run_parallel,
    "eager": run_eager,
}


# ---------------------------------------------------------------------------
# Live spot-check
# ---------------------------------------------------------------------------


async def run_live(provider: str, workload: Workload) -> float:
    """Spot-check against a real provider. Same workload tools, real LLM stream."""
    tools = build_tools(workload)
    if provider == "anthropic":
        from anthropic import AsyncAnthropic  # type: ignore[import-not-found]

        client = AsyncAnthropic()
        schemas = [
            {
                "name": t.name,
                "description": f"Stub tool {t.name}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            }
            for t in workload.tools
        ]
        prompt = (
            "Call ALL of these tools in parallel with q=\"x\" each: "
            + ", ".join(t.name for t in workload.tools)
        )
        t0 = time.perf_counter()
        async with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            tools=schemas,
            messages=[{"role": "user", "content": prompt}],
        ) as raw_stream:
            stream = AnthropicEagerStream(raw_stream, tools=tools)
            async for _ev in stream.events():
                pass
            async for _c, _r in stream.results():
                pass
        return (time.perf_counter() - t0) * 1000.0

    if provider == "openai":
        from eager_tools_openai import OpenAIEagerStream  # type: ignore[import-not-found]
        from openai import AsyncOpenAI  # type: ignore[import-not-found]

        client = AsyncOpenAI()
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": f"Stub tool {t.name}.",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
            }
            for t in workload.tools
        ]
        prompt = (
            "Call ALL of these tools in parallel with q=\"x\" each: "
            + ", ".join(t.name for t in workload.tools)
        )
        t0 = time.perf_counter()
        raw_stream: AsyncIterator[Any] = await client.chat.completions.create(
            model="gpt-4o-mini",
            tools=schemas,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        stream = OpenAIEagerStream(raw_stream, tools=tools)
        async for _ev in stream.events():
            pass
        async for _c, _r in stream.results():
            pass
        return (time.perf_counter() - t0) * 1000.0

    raise ValueError(f"unknown live provider: {provider}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def measure(workload: Workload, mode: str, runs: int) -> dict[str, float]:
    runner = MODES[mode]
    samples: list[float] = []
    for _ in range(runs):
        ms = await runner(workload)
        samples.append(ms)
    return {
        "p50": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Workload | Sequential p50 | Parallel p50 | **Eager p50** | Speedup vs parallel |",
        "|----------|----------------|--------------|---------------|---------------------|",
    ]
    for r in rows:
        seq = r["sequential"]["p50"] / 1000.0
        par = r["parallel"]["p50"] / 1000.0
        eag = r["eager"]["p50"] / 1000.0
        speedup = par / eag if eag > 0 else float("inf")
        lines.append(
            f"| {r['name']} | {seq:.1f}s | {par:.1f}s | **{eag:.1f}s** | {speedup:.1f}× |"
        )
    return "\n".join(lines)


def render_results_md(rows: list[dict[str, Any]], runs: int) -> str:
    return f"""# Bench results

> Generated by `make bench` (synthetic harness, deterministic).
> Reproduce: `make bench` from repo root.

## Headline (median of {runs} runs)

{render_table(rows)}

## Environment

- Python: {platform.python_version()}
- Platform: {platform.system()} {platform.release()} ({platform.machine()})

## What this measures

The harness paces a fake provider stream with realistic chunk timing
(`bench/harness.py`), then runs the *same* stream through three dispatch
modes:

- **sequential** — drain the stream fully, then run tools one at a time.
- **parallel** — drain the stream fully, then `asyncio.gather` all tools.
- **eager** — `AnthropicEagerStream` dispatches each tool the moment its
  JSON block seals, overlapping tool execution with the still-streaming model.

The synthetic stream removes network jitter so the comparison isolates the
dispatch strategy. For real-API spot-checks (subject to network/provider
variance), run `make bench-live`.

## Caveats

- Synthetic numbers are **lower bounds** on the real-world win — production
  streams have tail latency and jitter that compound the eager advantage.
- The "speedup vs parallel" column is the headline metric. Sequential is
  shown for context (it matches the pre-2024 default for many SDKs).
- Results vary by hardware and event-loop scheduling. Run on your own box
  before quoting these numbers.
"""


async def main_async(args: argparse.Namespace) -> None:
    if args.workload == "all":
        keys = list(WORKLOADS.keys())
    else:
        keys = [args.workload]

    if args.live:
        for k in keys:
            wl = WORKLOADS[k]
            print(f"→ live ({args.live}) {wl.name}")
            ms = await run_live(args.live, wl)
            print(f"   {ms / 1000.0:.2f}s")
        return

    rows: list[dict[str, Any]] = []
    for k in keys:
        wl = WORKLOADS[k]
        print(f"→ {wl.name}  ({len(wl.tools)} tools, ~{wl.stream_duration_ms / 1000:.0f}s stream)")
        row: dict[str, Any] = {"name": wl.name}
        for mode in ("sequential", "parallel", "eager"):
            row[mode] = await measure(wl, mode, args.runs)
            print(
                f"   {mode:11s} p50={row[mode]['p50'] / 1000:.2f}s  "
                f"(min={row[mode]['min'] / 1000:.2f}s, max={row[mode]['max'] / 1000:.2f}s)"
            )
        rows.append(row)

    print()
    print(render_table(rows))

    if not args.no_write:
        RESULTS_PATH.write_text(render_results_md(rows, args.runs))
        print(f"\nwrote {RESULTS_PATH.relative_to(REPO_ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=["all", "3", "8", "15"], default="all")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--live", choices=["anthropic", "openai"], default=None)
    p.add_argument("--no-write", action="store_true", help="don't write results.md")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
