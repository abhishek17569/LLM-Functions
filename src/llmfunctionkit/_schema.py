"""JSON Schema builder for llm_function.

Converts a Python function's signature and return annotation into JSON Schema
documents using Pydantic v2's :class:`TypeAdapter` under the hood. Descriptions
parsed from the docstring are merged into the generated input schema; the
``Returns:`` text is merged into the output schema.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from typing import Any, cast

from pydantic import TypeAdapter

from ._config import ConfigurationError
from ._docstring import DocstringSpec, parse_docstring

__all__ = [
    "build_input_schema",
    "build_output_schema",
]


def build_input_schema(
    fn: Callable[..., Any],
    *,
    docstring: DocstringSpec | None = None,
) -> dict[str, Any]:
    """Return a JSON Schema describing ``fn``'s parameters.

    Parameters with ``Field(description=...)`` defaults on Pydantic models are
    propagated. Descriptions parsed from the docstring's ``Args:`` section are
    merged into top-level properties only where Pydantic does not already have
    a description.
    """

    if docstring is None:
        docstring = parse_docstring(fn)
    sig = inspect.signature(fn)
    type_hints = _safe_get_type_hints(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = type_hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            raise ConfigurationError(
                f"Parameter {name!r} of {getattr(fn, '__qualname__', fn)!r} "
                "is missing a type annotation."
            )
        prop_schema = _schema_for(annotation, fn=fn, parameter=name)
        properties[name] = prop_schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {
        "title": f"{getattr(fn, '__name__', 'fn')}_input",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    _merge_arg_descriptions(schema, docstring.args)
    return schema


def build_output_schema(
    fn: Callable[..., Any],
    *,
    docstring: DocstringSpec | None = None,
) -> dict[str, Any]:
    """Return a JSON Schema for ``fn``'s return annotation.

    The ``Returns:`` text from the docstring becomes the schema's top-level
    ``description``.
    """

    if docstring is None:
        docstring = parse_docstring(fn)
    sig = inspect.signature(fn)
    type_hints = _safe_get_type_hints(fn)
    annotation = type_hints.get("return", sig.return_annotation)
    if annotation is inspect.Parameter.empty:
        raise ConfigurationError(
            f"{getattr(fn, '__qualname__', fn)!r} is missing a return type annotation."
        )
    if annotation is None or annotation is type(None):
        schema: dict[str, Any] = {"type": "null"}
    else:
        schema = _schema_for(annotation, fn=fn, parameter="<return>")

    if docstring.returns and "description" not in schema:
        schema["description"] = docstring.returns
    schema["title"] = f"{getattr(fn, '__name__', 'fn')}_output"
    return schema


def _safe_get_type_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        return typing.get_type_hints(fn, include_extras=True)
    except Exception:
        return {}


def _schema_for(annotation: Any, *, fn: Callable[..., Any], parameter: str) -> dict[str, Any]:
    try:
        adapter: TypeAdapter[Any] = TypeAdapter(annotation)
        raw = adapter.json_schema(ref_template="#/$defs/{model}")
    except Exception as exc:
        raise ConfigurationError(
            f"Cannot build JSON Schema for parameter {parameter!r} of "
            f"{getattr(fn, '__qualname__', fn)!r}: {exc}"
        ) from exc
    return _normalize_schema(raw)


def _normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$defs`` references so each schema fragment is self-contained."""

    defs = schema.pop("$defs", None) or schema.pop("definitions", None)
    inlined = _inline_refs(schema, defs or {}, set())
    if not isinstance(inlined, dict):
        return schema
    inlined.pop("$defs", None)
    return cast(dict[str, Any], inlined)


def _inline_refs(node: Any, defs: dict[str, Any], stack: set[str]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            ref = node["$ref"]
            key = ref.rsplit("/", 1)[-1]
            target = defs.get(key)
            if target is None or key in stack:
                return node
            new_stack = stack | {key}
            merged = _inline_refs(target, defs, new_stack)
            if isinstance(merged, dict):
                extras = {k: v for k, v in node.items() if k != "$ref"}
                out = dict(merged)
                out.update(_inline_refs(extras, defs, new_stack))
                return out
            return node
        return {k: _inline_refs(v, defs, stack) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(item, defs, stack) for item in node]
    return node


def _merge_arg_descriptions(schema: dict[str, Any], args: dict[str, str]) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    for name, description in args.items():
        prop = properties.get(name)
        if not isinstance(prop, dict):
            continue
        if not prop.get("description"):
            prop["description"] = description
