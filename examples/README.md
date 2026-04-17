# Examples

Runnable demonstrations of the eager-tools API.

| File | What it shows | Requires |
|------|---------------|----------|
| `01_minimal.py` | Eager dispatch end-to-end against an in-memory fake stream. No network. | nothing |
| `02_anthropic_live.py` | Real Claude stream + 3 fake tools. Watch tools fire as their JSON blocks seal. | `ANTHROPIC_API_KEY`, `pip install anthropic` |
| `03_openai_live.py` | Same harness against GPT-4o. | `OPENAI_API_KEY`, `pip install openai` |
| `04_cancellation.py` | Mid-flight `stream.cancel()` releases all in-flight tool tasks cleanly. | nothing |

## Running

From a venv with the adapter packages installed:

```bash
python examples/01_minimal.py
ANTHROPIC_API_KEY=sk-ant-... python examples/02_anthropic_live.py
OPENAI_API_KEY=sk-...        python examples/03_openai_live.py
python examples/04_cancellation.py
```

For development, you can use either adapter package's venv (each declares its own
`anthropic`/`openai` extra under `[project.optional-dependencies] dev`):

```bash
cd packages/eager-tools-anthropic && uv sync --extra dev
uv run python ../../examples/01_minimal.py
```

## What you should see

`01_minimal` and the live examples log a timeline like:

```
[   1.1 ms] stream opened — eager dispatch active
[ 102.0 ms] sealed fetch_weather   latency=100.8 ms — dispatched eagerly
[ 102.1 ms]   ↳ fetch_weather started
[ 203.4 ms] sealed fetch_stock     latency=101.3 ms — dispatched eagerly
...
```

The "sealed" lines arrive *during* the stream — well before `message_complete`.
Total wall clock ≈ duration of the slowest tool, not the sum of all tools.
