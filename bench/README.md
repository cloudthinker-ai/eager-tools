# bench/

Reproducible micro-benchmark for eager vs parallel vs sequential tool dispatch.

## Files

- `harness.py` — synthetic Anthropic-shaped stream + `LatencyTool`. Three
  workloads (3-tool, 8-tool, 15-tool) baked in.
- `run.py` — runner. CLI: `python bench/run.py [--workload all|3|8|15] [--runs N] [--live anthropic|openai]`.
- `results.md` — generated; checked in so you can read it without running.

## How to run

```bash
make bench                  # synthetic, deterministic, ~30s
make bench-live             # real Anthropic API (needs ANTHROPIC_API_KEY)
```

Or directly: `uv run --project packages/eager-tools-anthropic --with-editable packages/eager-tools-core python bench/run.py`.

## How to interpret

The synthetic harness paces a fake provider stream, then runs the same
stream through three dispatch modes. The **eager** mode overlaps tool
execution with the still-streaming model; the other two drain the stream
first.

- Speedup vs parallel is the headline metric — it isolates "what does
  eager dispatch buy me" from "what does parallel dispatch buy me".
- Synthetic numbers are a lower bound on the real-world win — production
  streams have tail latency that compounds the eager advantage. Use
  `--live` to spot-check.
- `--live` mode bills your API key. The model picks how many tools to
  call, so live numbers are noisier than synthetic.
