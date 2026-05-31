"""Pytest plugin that auto-promotes docstring ``Examples:`` blocks to tests.

For every ``@llm_function``-decorated callable in a collected module, each
``Examples:`` doctest entry with both an input and an output is registered as
a pytest item marked ``@pytest.mark.llm_function_example``. The item invokes
the function in ``cache="replay"`` mode and asserts equality with the
expected output parsed from the docstring.

Output parsing uses :func:`ast.literal_eval` only — never :func:`eval` — so
docstrings cannot smuggle arbitrary code into the test runner.

Activate with ``pytest -p llm_functions._doctest_runner`` or by adding
``llm_functions._doctest_runner`` to ``[tool.pytest.ini_options] plugins``.
Skip with ``-m "not llm_function_example"``.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ._docstring import ExampleSpec, parse_docstring

__all__ = [
    "ExampleCase",
    "LLMFunctionExampleItem",
    "collect_examples",
    "parse_example_input",
    "parse_example_output",
]


@dataclass(frozen=True)
class ExampleCase:
    """One executable example pulled from a docstring.

    Attributes:
        input_args: Positional arguments parsed from the doctest input line.
        input_kwargs: Keyword arguments parsed from the doctest input line.
        expected_output: Value parsed via :func:`ast.literal_eval` from the
            output line.
        raw_input: Original doctest input string, retained for error reports.
        raw_output: Original doctest output string, retained for error reports.
    """

    input_args: tuple[Any, ...]
    input_kwargs: dict[str, Any]
    expected_output: Any
    raw_input: str
    raw_output: str


def _is_ai_function(obj: Any) -> bool:
    """Return ``True`` if ``obj`` looks like an ``@llm_function``-decorated callable.

    The decorator is implemented by ``forge-decorator`` and not yet present
    when this module is imported, so we duck-type on the marker attribute
    rather than importing it. The decorator is expected to set
    ``__llm_function__ = True`` on the wrapper.
    """

    return callable(obj) and bool(getattr(obj, "__llm_function__", False))


def parse_example_input(raw: str) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Parse a doctest input like ``fn(1, b=2)`` into ``(args, kwargs)``.

    Returns ``None`` if the line is not a single function call expression or
    if any argument cannot be safely literal-evaluated.
    """

    text = raw.strip()
    if not text:
        return None
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    if not isinstance(tree, ast.Expression) or not isinstance(tree.body, ast.Call):
        return None
    call = tree.body
    args: list[Any] = []
    for node in call.args:
        if isinstance(node, ast.Starred):
            return None
        try:
            args.append(ast.literal_eval(node))
        except (ValueError, SyntaxError):
            return None
    kwargs: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            return None
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None
    return tuple(args), kwargs


def parse_example_output(raw: str) -> tuple[bool, Any]:
    """Parse a doctest output line via :func:`ast.literal_eval`.

    Returns ``(ok, value)``. ``ok`` is ``False`` when the line cannot be
    safely literal-evaluated; the runner will skip such examples rather than
    crash.
    """

    text = raw.strip()
    if not text:
        return False, None
    try:
        return True, ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return False, None


def _build_case(spec: ExampleSpec) -> ExampleCase | None:
    if not spec.input_repr or not spec.output_repr:
        return None
    parsed_in = parse_example_input(spec.input_repr)
    if parsed_in is None:
        return None
    ok, expected = parse_example_output(spec.output_repr)
    if not ok:
        return None
    args, kwargs = parsed_in
    return ExampleCase(
        input_args=args,
        input_kwargs=kwargs,
        expected_output=expected,
        raw_input=spec.input_repr,
        raw_output=spec.output_repr,
    )


def collect_examples(fn: Callable[..., Any]) -> list[ExampleCase]:
    """Return every well-formed :class:`ExampleCase` parsed from ``fn``'s docstring.

    Malformed entries (missing input or output, syntax errors, non-literal
    arguments) are silently dropped — the docstring parser preserves prose
    examples that do not represent executable cases.
    """

    spec = parse_docstring(fn)
    cases: list[ExampleCase] = []
    for example in spec.examples:
        case = _build_case(example)
        if case is not None:
            cases.append(case)
    return cases


def _iter_module_ai_functions(module: Any) -> Iterator[tuple[str, Callable[..., Any]]]:
    seen: set[int] = set()
    for name, obj in inspect.getmembers(module):
        if not _is_ai_function(obj):
            continue
        ident = id(obj)
        if ident in seen:
            continue
        seen.add(ident)
        yield name, obj


class LLMFunctionExampleItem(pytest.Item):
    """Pytest item that runs one :class:`ExampleCase` against an llm_function."""

    def __init__(
        self,
        *,
        name: str,
        parent: pytest.Collector,
        fn: Callable[..., Any],
        case: ExampleCase,
    ) -> None:
        super().__init__(name, parent)
        self._fn = fn
        self._case = case
        self.add_marker(pytest.mark.llm_function_example)

    def runtest(self) -> None:
        result = _run_in_replay_mode(self._fn, self._case)
        assert result == self._case.expected_output, (
            f"llm_function example mismatch for {self._fn.__qualname__}\n"
            f"  input:    {self._case.raw_input}\n"
            f"  expected: {self._case.raw_output}\n"
            f"  actual:   {result!r}"
        )

    def reportinfo(self) -> tuple[str | Path, int | None, str]:
        path = inspect.getsourcefile(self._fn) or "<unknown>"
        try:
            _, lineno = inspect.getsourcelines(self._fn)
        except (OSError, TypeError):
            lineno = 0
        return path, lineno, self.name


def _run_in_replay_mode(fn: Callable[..., Any], case: ExampleCase) -> Any:
    """Invoke ``fn`` with ``cache="replay"`` forced.

    The decorator (``forge-decorator``) is expected to honor a ``cache=``
    keyword. If it does not, we fall back to a direct call so that tests
    surface the misconfiguration via the assertion rather than a crash here.
    """

    try:
        return fn(*case.input_args, cache="replay", **case.input_kwargs)
    except TypeError:
        return fn(*case.input_args, **case.input_kwargs)


class _LLMFunctionModuleCollector(pytest.Module):
    """Wraps :class:`pytest.Module` to also yield llm_function example items."""

    def collect(self) -> Iterable[pytest.Item | pytest.Collector]:
        yield from super().collect()
        try:
            module = self.obj
        except Exception:
            return
        for fn_name, fn in _iter_module_ai_functions(module):
            for index, case in enumerate(collect_examples(fn)):
                item_name = f"{fn_name}[example{index}]"
                yield LLMFunctionExampleItem.from_parent(
                    parent=self, name=item_name, fn=fn, case=case
                )


def _module_imports_ai_function(path: Path) -> bool:
    """Return ``True`` if the source at ``path`` references ``llm_function``.

    A cheap heuristic used to keep collection lazy: most test modules don't
    use llm_function, so we don't want to import every collected file.
    """

    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "llm_function" in source


def _resolve_module(path: Path) -> Any | None:
    """Best-effort import of the module at ``path``. Returns ``None`` on failure."""

    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
    except (OSError, ValueError):
        return None
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "llm_function_example: auto-generated doctest example from an @llm_function docstring.",
    )


def pytest_collect_file(parent: pytest.Collector, file_path: Path) -> pytest.Collector | None:
    """Promote any Python file containing llm_function-decorated callables.

    Pytest's default collector handles ``test_*.py`` files; this hook only
    activates for non-test ``.py`` files that mention ``llm_function``.
    """

    if file_path.suffix != ".py":
        return None
    if file_path.name.startswith("test_") or file_path.name.endswith("_test.py"):
        return None
    if not _module_imports_ai_function(file_path):
        return None
    module = _resolve_module(file_path)
    if module is None:
        return None
    if not any(True for _ in _iter_module_ai_functions(module)):
        return None
    return _LLMFunctionModuleCollector.from_parent(parent=parent, path=file_path)
