"""eager-tools-anthropic — Anthropic adapter for eager tool calling.

Wraps `anthropic.AsyncMessageStream` (or any compatible async event iterator),
emits core `SealEvent`s the moment a tool block seals, and dispatches sealed
tools on a shared `ExecutorPool`.

See the repository `METHOD.md` for the mechanism and runtime contract.
"""

from __future__ import annotations

from .stream import AnthropicEagerStream

__version__ = "0.0.1"

__all__ = ["AnthropicEagerStream", "__version__"]
