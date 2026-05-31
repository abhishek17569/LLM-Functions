"""Tests for llm_functions._doctest_runner."""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from typing import Any, TypeVar

import pytest

from llm_functions._doctest_runner import (
    collect_examples,
    parse_example_input,
    parse_example_output,
)

pytest_plugins = ["pytester"]

F = TypeVar("F", bound=Callable[..., Any])


def _mark_ai(fn: F) -> F:
    """Stand-in for the real ``@llm_function`` decorator: just sets the marker."""

    fn.__llm_function__ = True  # type: ignore[attr-defined]
    return fn


# ---------------------------------------------------------------------------
# Three reference fixtures the runner is required to handle.
# ---------------------------------------------------------------------------


@_mark_ai
def fixture_add(a: int, b: int) -> int:
    """Add two numbers.

    Examples:
        >>> fixture_add(1, 2)
        3
        >>> fixture_add(10, -3)
        7
    """

    return a + b


@_mark_ai
def fixture_classify(text: str) -> dict[str, str]:
    """Classify text into a fixed taxonomy.

    Examples:
        >>> fixture_classify("hello")
        {'label': 'greeting'}
    """

    return {"label": "greeting"}


@_mark_ai
def fixture_unwrap(value: str) -> str:
    """Strip outer whitespace.

    Examples:
        >>> fixture_unwrap('  hi  ')
        'hi'
    """

    return value.strip()


@_mark_ai
def fixture_with_malformed(value: str) -> str:
    """Echo the input.

    Examples:
        >>> fixture_with_malformed('ok')
        'ok'
        >>> fixture_with_malformed(some_runtime_var)
        'unparseable'
        >>> fixture_with_malformed('only-input')
    """

    return value


# ---------------------------------------------------------------------------
# parse_example_input / parse_example_output unit tests.
# ---------------------------------------------------------------------------


def test_parse_input_simple_call() -> None:
    parsed = parse_example_input("fn(1, 2)")
    assert parsed == ((1, 2), {})


def test_parse_input_with_kwargs() -> None:
    parsed = parse_example_input("fn(1, b=2)")
    assert parsed == ((1,), {"b": 2})


def test_parse_input_rejects_starred_args() -> None:
    assert parse_example_input("fn(*args)") is None


def test_parse_input_rejects_double_starred_kwargs() -> None:
    assert parse_example_input("fn(**kwargs)") is None


def test_parse_input_rejects_non_call() -> None:
    assert parse_example_input("1 + 2") is None


def test_parse_input_rejects_non_literal_args() -> None:
    # ``some_var`` is a Name node; literal_eval refuses it.
    assert parse_example_input("fn(some_var)") is None


def test_parse_input_rejects_syntax_errors() -> None:
    assert parse_example_input("fn(") is None


def test_parse_input_handles_blank_input() -> None:
    assert parse_example_input("   ") is None


def test_parse_output_succeeds_on_literal() -> None:
    ok, value = parse_example_output("[1, 2, 3]")
    assert ok and value == [1, 2, 3]


def test_parse_output_fails_on_non_literal() -> None:
    ok, _ = parse_example_output("some_var")
    assert ok is False


def test_parse_output_fails_on_blank() -> None:
    ok, _ = parse_example_output("")
    assert ok is False


# ---------------------------------------------------------------------------
# collect_examples on the three reference fixtures.
# ---------------------------------------------------------------------------


def test_collect_examples_for_add() -> None:
    cases = collect_examples(fixture_add)
    assert len(cases) == 2
    assert cases[0].input_args == (1, 2)
    assert cases[0].expected_output == 3
    assert cases[1].input_args == (10, -3)
    assert cases[1].expected_output == 7


def test_collect_examples_for_classify() -> None:
    cases = collect_examples(fixture_classify)
    assert len(cases) == 1
    assert cases[0].input_args == ("hello",)
    assert cases[0].expected_output == {"label": "greeting"}


def test_collect_examples_for_unwrap() -> None:
    cases = collect_examples(fixture_unwrap)
    assert len(cases) == 1
    assert cases[0].input_args == ("  hi  ",)
    assert cases[0].expected_output == "hi"


def test_collect_examples_drops_malformed_entries() -> None:
    cases = collect_examples(fixture_with_malformed)
    # Only the first example is well-formed; the runtime-var input and the
    # missing-output case are both dropped silently.
    assert len(cases) == 1
    assert cases[0].input_args == ("ok",)
    assert cases[0].expected_output == "ok"


def test_collect_examples_returns_empty_for_undocumented_fn() -> None:
    @_mark_ai
    def no_docstring(x: int) -> int:
        return x

    assert collect_examples(no_docstring) == []


# ---------------------------------------------------------------------------
# LLMFunctionExampleItem assertion behaviour.
# ---------------------------------------------------------------------------


def test_pytest_collects_llm_function_examples_in_module(pytester: pytest.Pytester) -> None:
    """End-to-end: pytest run actually picks up auto-generated example items."""

    pytester.makepyfile(
        my_ai_module=textwrap.dedent(
            '''
            """Module that exposes one llm_function-style callable."""

            def _mark(fn):
                fn.__llm_function__ = True
                return fn

            @_mark
            def adder(a, b, cache=None):
                """Add.

                Examples:
                    >>> adder(2, 3)
                    5
                """
                return a + b
            '''
        )
    )
    pytester.makepyfile(
        conftest=textwrap.dedent(
            """
            pytest_plugins = ["llm_functions._doctest_runner"]
            """
        )
    )
    result = pytester.runpytest("-q", "--co")
    output = "\n".join(result.outlines)
    assert "adder[example0]" in output


def test_pytest_runs_collected_example_assertion(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        my_ok_module=textwrap.dedent(
            """
            def _mark(fn):
                fn.__llm_function__ = True
                return fn

            @_mark
            def echo(x, cache=None):
                '''Echo.

                Examples:
                    >>> echo(7)
                    7
                '''
                return x
            """
        )
    )
    pytester.makepyfile(
        conftest=textwrap.dedent(
            """
            pytest_plugins = ["llm_functions._doctest_runner"]
            """
        )
    )
    result = pytester.runpytest("-q")
    assert result.ret == 0
    result.assert_outcomes(passed=1)


def test_pytest_failing_example_fails_test(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        my_bad_module=textwrap.dedent(
            """
            def _mark(fn):
                fn.__llm_function__ = True
                return fn

            @_mark
            def wrong(x, cache=None):
                '''Wrong.

                Examples:
                    >>> wrong(1)
                    99
                '''
                return x
            """
        )
    )
    pytester.makepyfile(
        conftest=textwrap.dedent(
            """
            pytest_plugins = ["llm_functions._doctest_runner"]
            """
        )
    )
    result = pytester.runpytest("-q")
    assert result.ret != 0
    result.assert_outcomes(failed=1)


def test_marker_lets_users_skip_examples(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        my_skip_module=textwrap.dedent(
            """
            def _mark(fn):
                fn.__llm_function__ = True
                return fn

            @_mark
            def echo(x, cache=None):
                '''Echo.

                Examples:
                    >>> echo(1)
                    1
                '''
                return x
            """
        )
    )
    pytester.makepyfile(
        conftest=textwrap.dedent(
            """
            pytest_plugins = ["llm_functions._doctest_runner"]
            """
        )
    )
    result = pytester.runpytest("-q", "-m", "not llm_function_example")
    # No tests selected — pytest may exit with code 5 (no tests collected) but
    # must not fail any auto-generated example.
    output = "\n".join(result.outlines)
    assert "failed" not in output


def test_collector_skips_files_without_ai_function_reference(
    pytester: pytest.Pytester,
) -> None:
    """Non-test .py files that don't reference llm_function are not collected."""

    pytester.makepyfile(
        unrelated=textwrap.dedent(
            """
            def hello():
                return 'world'
            """
        )
    )
    pytester.makepyfile(
        conftest=textwrap.dedent(
            """
            pytest_plugins = ["llm_functions._doctest_runner"]
            """
        )
    )
    result = pytester.runpytest("-q", "--co")
    output = "\n".join(result.outlines)
    assert "unrelated" not in output


def test_collector_falls_back_to_direct_call_when_cache_kwarg_unsupported(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(
        my_no_cache_kwarg=textwrap.dedent(
            """
            def _mark(fn):
                fn.__llm_function__ = True
                return fn

            @_mark
            def add(a, b):
                '''Add.

                Examples:
                    >>> add(2, 3)
                    5
                '''
                return a + b
            """
        )
    )
    pytester.makepyfile(
        conftest=textwrap.dedent(
            """
            pytest_plugins = ["llm_functions._doctest_runner"]
            """
        )
    )
    result = pytester.runpytest("-q")
    result.assert_outcomes(passed=1)
