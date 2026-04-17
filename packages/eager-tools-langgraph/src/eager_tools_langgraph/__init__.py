"""LangGraph adapter for eager tool calling.

Public surface:

- `EagerMiddleware` — `AgentMiddleware` subclass that owns the model stream and
  dispatches idempotent tool calls the moment they seal.
- `eager_middleware(tools, **kwargs)` — convenience factory.

See README for the version matrix and the 5 honest limits.
"""

from __future__ import annotations

from .middleware import EagerMiddleware, eager_middleware

__all__ = ["EagerMiddleware", "eager_middleware"]
