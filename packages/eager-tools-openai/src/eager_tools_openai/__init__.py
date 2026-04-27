"""eager-tools-openai — OpenAI adapter for eager tool calling.

Wraps an OpenAI `chat.completions` streaming iterator, emits core `SealEvent`s
the moment a tool block seals, and dispatches sealed tools on a shared
`ExecutorPool`.

See the repository `METHOD.md` for the mechanism and runtime contract.
"""

from __future__ import annotations

from .stream import OpenAIEagerStream

__version__ = "0.3.0"

__all__ = ["OpenAIEagerStream", "__version__"]
