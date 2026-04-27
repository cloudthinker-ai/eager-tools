"""eager-tools-core — provider-agnostic core for eager tool calling.

Public API:

    from eager_tools import (
        SealDetector, ExecutorPool,
        ToolCall, SealEvent, Tool, ObservabilityHook,
        EagerDispatchDeniedError, NonIdempotentToolError, GateDeniedError, GateFn,
    )

See the repository `METHOD.md` for the mechanism and runtime contract.
"""

from __future__ import annotations

from .core import SealDetector
from .executor import (
    EagerDispatchDeniedError,
    ExecutorPool,
    GateDeniedError,
    NonIdempotentToolError,
)
from .observability import OTelObservability
from .types import (
    NOOP_OBSERVABILITY,
    GateFn,
    ObservabilityHook,
    SealEvent,
    SealKind,
    Tool,
    ToolCall,
)

__version__ = "0.3.0"

__all__ = [
    "NOOP_OBSERVABILITY",
    "EagerDispatchDeniedError",
    "ExecutorPool",
    "GateDeniedError",
    "GateFn",
    "NonIdempotentToolError",
    "OTelObservability",
    "ObservabilityHook",
    "SealDetector",
    "SealEvent",
    "SealKind",
    "Tool",
    "ToolCall",
    "__version__",
]
