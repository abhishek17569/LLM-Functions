"""Tool registry and dispatcher for llm_function.

A tool is any Python callable the LLM can invoke during a completion. Each
tool gets a JSON-Schema definition built from its signature; the dispatcher
validates incoming arguments before invocation and re-raises any exception
the tool throws as :class:`ToolExecutionError`.

The ``@llm_tool`` decorator is intentionally lightweight: it simply tags a
callable with ``__ai_tool__ = True`` and returns it unchanged. Authors who
want to expose a function as a tool decorate it with ``@llm_tool``; callers
add it to the ``tools=[...]`` list on ``@llm_function``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from ._config import ConfigurationError, Settings
from ._exceptions import ToolExecutionError
from ._schema import build_input_schema

__all__ = [
    "ToolDef",
    "ToolRegistry",
    "invoke_tool",
    "llm_tool",
    "resolve_tools",
]


@dataclass(frozen=True)
class ToolDef:
    """A registered tool: the original callable plus its OpenAI/LiteLLM schema."""

    name: str
    description: str
    fn: Callable[..., Any]
    schema: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        """Return the OpenAI/LiteLLM-style ``tools[]`` entry for this tool."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


def llm_tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Mark ``fn`` as available for use as an llm_function tool.

    Supports both ``@llm_tool`` and ``@llm_tool(name="x")``. The wrapped object
    is returned unchanged save for two new attributes (``__ai_tool__``,
    ``__ai_tool_meta__``) so direct ``fn(...)`` calls remain transparent.
    """

    def _apply(target: Callable[..., Any]) -> Callable[..., Any]:
        target.__ai_tool__ = True  # type: ignore[attr-defined]
        target.__ai_tool_meta__ = {  # type: ignore[attr-defined]
            "name": name or target.__name__,
            "description": description,
        }
        return target

    if fn is not None:
        return _apply(fn)
    return _apply


class ToolRegistry:
    """In-memory registry of callable tools, keyed by their public name."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> ToolDef:
        """Register ``fn`` and return its :class:`ToolDef`.

        ``name`` defaults to ``fn.__name__``; ``description`` defaults to the
        first line of ``fn``'s docstring.
        """

        meta = getattr(fn, "__ai_tool_meta__", {}) or {}
        resolved_name = name or meta.get("name") or fn.__name__
        resolved_description = (
            description
            or meta.get("description")
            or _first_docstring_line(fn)
            or f"Invoke {resolved_name}."
        )
        schema = build_input_schema(fn)
        tool_def = ToolDef(
            name=resolved_name,
            description=resolved_description,
            fn=fn,
            schema=schema,
        )
        self._tools[resolved_name] = tool_def
        return tool_def

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def all_tools(self) -> list[ToolDef]:
        return list(self._tools.values())


_GLOBAL_REGISTRY: ToolRegistry = ToolRegistry()


def _first_docstring_line(fn: Callable[..., Any]) -> str | None:
    doc = inspect.getdoc(fn)
    if not doc:
        return None
    return doc.splitlines()[0].strip() or None


def resolve_tools(
    allowlist: list[Callable[..., Any]] | Literal["*"] | None,
    settings: Settings,
) -> list[ToolDef]:
    """Resolve a per-call tool allowlist to a list of :class:`ToolDef`.

    * ``None`` → no tools.
    * ``"*"`` → every tool ever registered via ``@llm_tool``. Requires
      ``Settings.allow_all_tools=True``; otherwise :class:`ConfigurationError`.
    * a list of callables → each is registered (idempotently) into a
      throwaway registry so ad-hoc functions are accepted, and each
      :class:`ToolDef` is returned in input order.
    """

    if allowlist is None:
        return []

    if allowlist == "*":
        if not settings.allow_all_tools:
            raise ConfigurationError(
                "tools='*' requires configure(allow_all_tools=True). "
                "Be explicit about which tools are exposed to the model."
            )
        return _GLOBAL_REGISTRY.all_tools()

    if not isinstance(allowlist, list):
        raise ConfigurationError(
            f"tools= must be a list of callables or the string '*'; got {type(allowlist).__name__}"
        )

    registry = ToolRegistry()
    resolved: list[ToolDef] = []
    for fn in allowlist:
        if not callable(fn):
            raise ConfigurationError(
                f"tools= contained a non-callable entry of type {type(fn).__name__}"
            )
        resolved.append(registry.register(fn))
        # Also register in the global registry so ``tools="*"`` picks it up
        # later. Idempotent — same name overwrites the earlier definition.
        _GLOBAL_REGISTRY.register(fn)
    return resolved


async def invoke_tool(
    name: str,
    args: dict[str, Any],
    allowed_tools: list[ToolDef],
) -> Any:
    """Dispatch a tool call by name against ``allowed_tools``.

    Validates ``args`` against the tool schema before calling. Both sync and
    async callables are supported. Any exception raised by the tool body is
    wrapped in :class:`ToolExecutionError` with the original as ``__cause__``.
    """

    tool = _find_tool(name, allowed_tools)
    if tool is None:
        raise ToolExecutionError(
            f"Model invoked unknown tool {name!r}. "
            f"Allowed tools: {[t.name for t in allowed_tools] or '<none>'}"
        )

    _validate_args(tool, args)

    try:
        result = tool.fn(**args)
        if inspect.isawaitable(result):
            result = await result
    except ToolExecutionError:
        raise
    except BaseException as exc:
        # _LLMFunctionRaisedException must propagate untouched — it's a
        # BaseException-only sentinel that the decorator boundary catches.
        from ._exceptions import _LLMFunctionRaisedException

        if isinstance(exc, _LLMFunctionRaisedException):
            raise
        if not isinstance(exc, Exception):
            raise
        wrapper = ToolExecutionError(f"tool {name!r} raised {type(exc).__name__}: {exc}")
        raise wrapper from exc
    return result


def _find_tool(name: str, allowed: list[ToolDef]) -> ToolDef | None:
    for tool in allowed:
        if tool.name == name:
            return tool
    return None


def _validate_args(tool: ToolDef, args: dict[str, Any]) -> None:
    """Lightweight validation against the tool's JSON Schema.

    A full JSON-Schema validator is overkill for the small subset we generate
    here; instead we enforce the two invariants that actually matter for
    safety: required keys are present, and unknown keys are rejected when
    ``additionalProperties: False``.
    """

    properties = cast(dict[str, Any], tool.schema.get("properties") or {})
    required = cast(list[str], tool.schema.get("required") or [])
    additional = tool.schema.get("additionalProperties", True)

    missing = [key for key in required if key not in args]
    if missing:
        raise ToolExecutionError(f"tool {tool.name!r} missing required argument(s): {missing}")

    if additional is False:
        unknown = [key for key in args if key not in properties]
        if unknown:
            raise ToolExecutionError(f"tool {tool.name!r} received unknown argument(s): {unknown}")
