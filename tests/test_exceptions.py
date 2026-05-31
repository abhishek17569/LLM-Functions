"""Tests for llmfunctionkit._exceptions."""

from __future__ import annotations

from llmfunctionkit._docstring import RaiseSpec
from llmfunctionkit._exceptions import (
    ConfigurationError,
    LLMFunctionError,
    OutputValidationError,
    ProviderError,
    ReplayMissError,
    ToolExecutionError,
    ToolIterationError,
    _LLMFunctionRaisedException,
    synthesise_raise_tools,
)


class _CustomError(Exception):
    pass


def test_synthesise_raise_tools_emits_one_tool_per_spec() -> None:
    raises = [
        RaiseSpec(name="ValueError", when="input is bad", exc_type=ValueError),
        RaiseSpec(name="KeyError", when="missing key", exc_type=KeyError),
    ]
    tools = synthesise_raise_tools(raises)
    assert [t["function"]["name"] for t in tools] == ["raise_ValueError", "raise_KeyError"]


def test_synthesise_raise_tools_schema_shape() -> None:
    raises = [RaiseSpec(name="ValueError", when="bad input", exc_type=ValueError)]
    [tool] = synthesise_raise_tools(raises)
    fn = tool["function"]
    assert fn["parameters"]["type"] == "object"
    assert "reason" in fn["parameters"]["properties"]
    assert fn["parameters"]["required"] == ["reason"]
    assert fn["parameters"]["additionalProperties"] is False
    assert "bad input" in fn["description"]


def test_synthesise_raise_tools_handles_empty_when() -> None:
    raises = [RaiseSpec(name="RuntimeError", when="", exc_type=RuntimeError)]
    [tool] = synthesise_raise_tools(raises)
    desc = tool["function"]["description"]
    assert "RuntimeError" in desc


def test_synthesise_raise_tools_empty_input() -> None:
    assert synthesise_raise_tools([]) == []


def test_raised_exception_carries_type_and_reason() -> None:
    raised = _LLMFunctionRaisedException(_CustomError, "because reasons")
    assert raised.exc_type is _CustomError
    assert raised.reason == "because reasons"
    assert "_CustomError" in str(raised)


def test_public_exceptions_are_aifunctionerror_subclasses() -> None:
    for exc in (
        ConfigurationError,
        OutputValidationError,
        ProviderError,
        ReplayMissError,
        ToolExecutionError,
        ToolIterationError,
    ):
        assert issubclass(exc, LLMFunctionError)
