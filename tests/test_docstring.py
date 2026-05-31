"""Fixtures across Google, Sphinx, and NumPy docstring styles."""

from __future__ import annotations

import pytest

from llmfunctionkit._config import ConfigurationError
from llmfunctionkit._docstring import DocstringSpec, parse_docstring


class CustomDomainError(Exception):
    """Domain-specific error used to verify global resolution of Raises."""


def test_google_full_sections() -> None:
    def fn(name: str, retries: int = 3) -> str:
        """Summarize an article.

        Some lead-in context describing intent.

        Args:
            name: The article slug.
            retries: How many times to retry on transient failure.

        Returns:
            A short summary string.

        Raises:
            ValueError: If ``name`` is empty.
            CustomDomainError: If the article cannot be found.

        Examples:
            >>> fn("hello")
            'Hello!'

        Constraints:
            - Output must be under 200 chars.
            - No PII allowed.

        Tools:
            search_web: Use this for fresh news only.

        Format:
            One paragraph, no markdown.

        Notes:
            Cached aggressively; uses the cheapest model by default.

        Model: openai/gpt-4o-mini
        Cache: on
        """

    spec = parse_docstring(fn)
    assert "Summarize an article." in spec.task
    assert spec.args == {
        "name": "The article slug.",
        "retries": "How many times to retry on transient failure.",
    }
    assert spec.returns is not None and "summary" in spec.returns
    raise_names = [r.name for r in spec.raises]
    assert raise_names == ["ValueError", "CustomDomainError"]
    assert spec.raises[0].exc_type is ValueError
    assert spec.raises[1].exc_type is CustomDomainError
    assert spec.examples and spec.examples[0].input_repr == 'fn("hello")'
    assert spec.examples[0].output_repr == "'Hello!'"
    assert "Output must be under 200 chars." in spec.constraints
    assert spec.tools_guidance == {"search_web": "Use this for fresh news only."}
    assert spec.format_hints == "One paragraph, no markdown."
    assert spec.notes is not None and "Cached aggressively" in spec.notes
    assert spec.model == "openai/gpt-4o-mini"
    assert spec.cache == "on"


def test_sphinx_style() -> None:
    def fn(name: str, retries: int = 3) -> str:
        """Summarize an article.

        :param name: The article slug.
        :param retries: How many times to retry on transient failure.
        :returns: A short summary string.
        :raises ValueError: If ``name`` is empty.
        :raises CustomDomainError: If the article cannot be found.
        """

    spec = parse_docstring(fn)
    assert spec.task.startswith("Summarize an article.")
    assert spec.args["name"] == "The article slug."
    assert spec.args["retries"] == "How many times to retry on transient failure."
    assert spec.returns == "A short summary string."
    assert [r.name for r in spec.raises] == ["ValueError", "CustomDomainError"]
    assert spec.raises[1].exc_type is CustomDomainError


def test_numpy_style() -> None:
    def fn(x: int, y: int) -> int:
        """Add two numbers.

        Parameters
        ----------
        x : int
            The first addend.
        y : int
            The second addend.

        Returns
        -------
        int
            The sum.

        Raises
        ------
        ValueError
            If inputs are negative.
        """

    spec = parse_docstring(fn)
    assert spec.task.startswith("Add two numbers.")
    assert spec.args == {"x": "The first addend.", "y": "The second addend."}
    assert spec.returns == "The sum."
    assert [r.name for r in spec.raises] == ["ValueError"]
    assert spec.raises[0].exc_type is ValueError


def test_single_line_docstring() -> None:
    def fn() -> None:
        """Just a plain one-liner."""

    spec = parse_docstring(fn)
    assert spec.task == "Just a plain one-liner."
    assert spec.args == {}
    assert spec.returns is None


def test_no_docstring_returns_empty_spec() -> None:
    def fn() -> None: ...

    spec = parse_docstring(fn)
    assert spec == DocstringSpec()


def test_aliases_are_case_insensitive_and_singular() -> None:
    def fn(a: int) -> int:
        """Lead-in.

        param:
            a: alpha
        Return:
            stuff
        Raise:
            ValueError: bad input
        """

    spec = parse_docstring(fn)
    assert spec.args == {"a": "alpha"}
    assert spec.returns == "stuff"
    assert spec.raises and spec.raises[0].name == "ValueError"


def test_unknown_section_goes_to_extra() -> None:
    def fn() -> None:
        """Lead-in.

        Mystery:
            who knows what this is

        Cache: off
        """

    spec = parse_docstring(fn)
    assert "Mystery" in spec.extra
    assert "who knows what this is" in spec.extra["Mystery"]
    assert spec.cache == "off"


def test_invalid_cache_value_raises() -> None:
    def fn() -> None:
        """Bad cache.

        Cache: maybe
        """

    with pytest.raises(ConfigurationError):
        parse_docstring(fn)


def test_unresolvable_exception_raises() -> None:
    def fn() -> None:
        """Bad raises.

        Raises:
            NotARealExceptionXYZ: never resolves
        """

    with pytest.raises(ConfigurationError):
        parse_docstring(fn)


def test_doctest_and_prose_examples_split() -> None:
    def fn() -> None:
        """Lead.

        Examples:
            Some prose explaining the example.

            >>> fn()
            42
        """

    spec = parse_docstring(fn)
    assert len(spec.examples) == 2
    prose = spec.examples[0]
    doctest = spec.examples[1]
    assert prose.prose is not None and "Some prose" in prose.prose
    assert doctest.input_repr == "fn()"
    assert doctest.output_repr == "42"
