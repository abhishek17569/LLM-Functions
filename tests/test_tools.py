"""Tests for llm_functions._tools."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from llm_functions._config import (
    ConfigurationError,
    Settings,
    configure,
    reset_settings,
)
from llm_functions._exceptions import ToolExecutionError
from llm_functions._tools import (
    ToolDef,
    ToolRegistry,
    invoke_tool,
    llm_tool,
    resolve_tools,
)


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    reset_settings()
    yield
    reset_settings()


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "model": "openai/gpt-4o-mini",
        "cache": "off",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_ai_tool_marks_callable_idempotently() -> None:
    @llm_tool
    def add(x: int, y: int) -> int:
        """Add two integers."""
        return x + y

    assert getattr(add, "__ai_tool__", False) is True
    assert add.__ai_tool_meta__["name"] == "add"  # type: ignore[attr-defined]
    assert add(2, 3) == 5


def test_ai_tool_with_explicit_name_and_description() -> None:
    @llm_tool(name="custom_name", description="custom desc")
    def helper(x: int) -> int:
        return x

    assert helper.__ai_tool_meta__["name"] == "custom_name"  # type: ignore[attr-defined]
    assert helper.__ai_tool_meta__["description"] == "custom desc"  # type: ignore[attr-defined]


def test_registry_registers_callable_with_schema() -> None:
    def fetch(query: str, limit: int = 5) -> list[str]:
        """Search the catalog."""
        return [query] * limit

    registry = ToolRegistry()
    tool_def = registry.register(fetch)
    assert isinstance(tool_def, ToolDef)
    assert tool_def.name == "fetch"
    assert tool_def.description.startswith("Search")
    assert "query" in tool_def.schema["properties"]
    assert tool_def.schema["properties"]["query"]["type"] == "string"


def test_resolve_tools_none_returns_empty_list() -> None:
    assert resolve_tools(None, _settings()) == []


def test_resolve_tools_list_returns_in_order() -> None:
    @llm_tool
    def alpha(x: int) -> int:
        """A."""
        return x

    @llm_tool
    def beta(y: int) -> int:
        """B."""
        return y

    resolved = resolve_tools([alpha, beta], _settings())
    assert [t.name for t in resolved] == ["alpha", "beta"]


def test_resolve_tools_star_requires_allow_all_tools() -> None:
    with pytest.raises(ConfigurationError, match="allow_all_tools"):
        resolve_tools("*", _settings(allow_all_tools=False))


def test_resolve_tools_star_works_when_allowed() -> None:
    configure(allow_all_tools=True)

    @llm_tool
    def globally_visible(x: int) -> int:
        """G."""
        return x

    # Register via resolve_tools so it lands in the global registry.
    resolve_tools([globally_visible], _settings(allow_all_tools=True))
    resolved = resolve_tools("*", _settings(allow_all_tools=True))
    names = [t.name for t in resolved]
    assert "globally_visible" in names


def test_resolve_tools_rejects_non_callable() -> None:
    with pytest.raises(ConfigurationError, match="non-callable"):
        resolve_tools([42], _settings())  # type: ignore[list-item]


def test_resolve_tools_rejects_invalid_type() -> None:
    with pytest.raises(ConfigurationError, match="must be a list"):
        resolve_tools({"alpha"}, _settings())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_invoke_tool_dispatches_sync() -> None:
    def add(x: int, y: int) -> int:
        """Add."""
        return x + y

    tools = resolve_tools([add], _settings())
    assert await invoke_tool("add", {"x": 1, "y": 2}, tools) == 3


@pytest.mark.asyncio
async def test_invoke_tool_dispatches_async() -> None:
    async def fetch(name: str) -> str:
        """Async fetch."""
        return f"hello {name}"

    tools = resolve_tools([fetch], _settings())
    assert await invoke_tool("fetch", {"name": "world"}, tools) == "hello world"


@pytest.mark.asyncio
async def test_invoke_tool_unknown_name_errors() -> None:
    with pytest.raises(ToolExecutionError, match="unknown tool"):
        await invoke_tool("nope", {}, [])


@pytest.mark.asyncio
async def test_invoke_tool_missing_required_arg_errors() -> None:
    def add(x: int, y: int) -> int:
        """Add."""
        return x + y

    tools = resolve_tools([add], _settings())
    with pytest.raises(ToolExecutionError, match="missing required"):
        await invoke_tool("add", {"x": 1}, tools)


@pytest.mark.asyncio
async def test_invoke_tool_unknown_arg_errors() -> None:
    def square(x: int) -> int:
        """Square."""
        return x * x

    tools = resolve_tools([square], _settings())
    with pytest.raises(ToolExecutionError, match="unknown argument"):
        await invoke_tool("square", {"x": 2, "extra": True}, tools)


@pytest.mark.asyncio
async def test_invoke_tool_wraps_user_exception() -> None:
    def boom(x: int) -> int:
        """B."""
        raise RuntimeError("nope")

    tools = resolve_tools([boom], _settings())
    with pytest.raises(ToolExecutionError) as ei:
        await invoke_tool("boom", {"x": 1}, tools)
    assert isinstance(ei.value.__cause__, RuntimeError)


def test_tool_def_to_openai_shape() -> None:
    def helper(name: str) -> str:
        """One-liner."""
        return name

    [tool_def] = resolve_tools([helper], _settings())
    payload = tool_def.to_openai()
    assert payload["type"] == "function"
    assert payload["function"]["name"] == "helper"
    assert payload["function"]["parameters"]["type"] == "object"
