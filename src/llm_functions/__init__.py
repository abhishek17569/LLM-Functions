"""llm_function — decorator-based LLM-native typed Python functions.

Top-level public surface. Most users only need :func:`llm_function`,
:func:`llm_tool`, and :func:`configure`; the exception types are exported for
``except`` blocks.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._config import ProviderConfig, configure
from ._decorator import llm_function
from ._exceptions import (
    ConfigurationError,
    LLMFunctionError,
    OutputValidationError,
    ProviderError,
    ReplayMissError,
    ToolExecutionError,
    ToolIterationError,
)
from ._tools import llm_tool

__all__ = [
    "ConfigurationError",
    "LLMFunctionError",
    "OutputValidationError",
    "ProviderConfig",
    "ProviderError",
    "ReplayMissError",
    "ToolExecutionError",
    "ToolIterationError",
    "__version__",
    "configure",
    "llm_function",
    "llm_tool",
]

try:
    __version__: str = version("llm-functions")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"
