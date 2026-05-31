"""Public exception surface for llm_function plus dynamic exception synthesis.

Re-exports the exception types defined in lower foundation modules so callers
have a single import (`from llmfunctionkit import …`) for every error.

Also defines:

* :func:`synthesise_raise_tools` — turns each ``RaiseSpec`` parsed from a
  docstring into a JSON-Schema tool definition the model can invoke to signal
  "I cannot complete this; raise X with this reason."
* :class:`_LLMFunctionRaisedException` — internal sentinel carried out of the
  tool dispatcher and converted by the decorator boundary into the
  user-declared exception.
"""

from __future__ import annotations

from typing import Any

from ._config import ConfigurationError, LLMFunctionError
from ._docstring import RaiseSpec
from ._provider import OutputValidationError, ProviderError, ToolIterationError
from ._replay import ReplayMissError

__all__ = [
    "ConfigurationError",
    "LLMFunctionError",
    "OutputValidationError",
    "ProviderError",
    "ReplayMissError",
    "ToolExecutionError",
    "ToolIterationError",
    "synthesise_raise_tools",
]


class ToolExecutionError(LLMFunctionError):
    """Raised when a user-supplied tool raises while being invoked.

    The original exception is attached as ``__cause__``.
    """


class _LLMFunctionRaisedException(BaseException):
    """Internal sentinel used to surface a docstring-declared exception.

    Subclasses :class:`BaseException` (not :class:`Exception`) to reduce the
    chance of being caught by user-tool ``except Exception`` handlers running
    inside the dispatch loop.
    """

    def __init__(self, exc_type: type[BaseException], reason: str) -> None:
        super().__init__(f"{exc_type.__name__}: {reason}")
        self.exc_type = exc_type
        self.reason = reason


def synthesise_raise_tools(raises: list[RaiseSpec]) -> list[dict[str, Any]]:
    """Build LiteLLM/OpenAI-style tool definitions for each declared exception.

    Each tool is named ``raise_<ExcName>`` and accepts a single ``reason``
    string. The model invokes one when it cannot satisfy the function's
    contract; the decorator's tool dispatcher converts the call into the
    declared Python exception.
    """

    tools: list[dict[str, Any]] = []
    for raise_spec in raises:
        description = (
            f"Signal that the function should raise {raise_spec.name}. Use when: {raise_spec.when}"
            if raise_spec.when
            else f"Signal that the function should raise {raise_spec.name}."
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"raise_{raise_spec.name}",
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": (
                                    "A short human-readable explanation that will be passed "
                                    f"to {raise_spec.name} as the exception message."
                                ),
                            },
                        },
                        "required": ["reason"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools
