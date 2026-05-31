"""Docstring parser for llm_function.

Parses Google, Sphinx (``:param x:``), and NumPy (``Parameters\\n----------``)
styles into a :class:`DocstringSpec` dataclass. Section headers are
case-insensitive and accept aliases (``Return``/``Returns``, ``Raise``/
``Raises``, ``Arg``/``Args``/``Param``/``Params``/``Parameter``/
``Parameters``).
"""

from __future__ import annotations

import builtins
import inspect
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from ._config import ConfigurationError

__all__ = [
    "DocstringSpec",
    "ExampleSpec",
    "RaiseSpec",
    "parse_docstring",
]


CacheMode = Literal["on", "off", "replay"]


@dataclass
class RaiseSpec:
    """One ``Raises:`` entry, with the resolved exception type."""

    name: str
    when: str
    exc_type: type[BaseException]


@dataclass
class ExampleSpec:
    """One ``Examples:`` entry. Doctest blocks are split into input/output.

    Free-form prose is captured in :attr:`prose`.
    """

    input_repr: str | None = None
    output_repr: str | None = None
    prose: str | None = None


@dataclass
class DocstringSpec:
    """Structured representation of a parsed docstring."""

    task: str = ""
    context: str | None = None
    args: dict[str, str] = field(default_factory=dict)
    returns: str | None = None
    raises: list[RaiseSpec] = field(default_factory=list)
    examples: list[ExampleSpec] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    tools_guidance: dict[str, str] = field(default_factory=dict)
    format_hints: str | None = None
    notes: str | None = None
    model: str | None = None
    cache: CacheMode | None = None
    extra: dict[str, str] = field(default_factory=dict)


_ARG_ALIASES = {"arg", "args", "param", "params", "parameter", "parameters"}
_RETURN_ALIASES = {"return", "returns"}
_RAISE_ALIASES = {"raise", "raises"}
_EXAMPLE_ALIASES = {"example", "examples"}
_CONSTRAINT_ALIASES = {"constraint", "constraints"}
_TOOLS_ALIASES = {"tool", "tools"}
_FORMAT_ALIASES = {"format", "format hints"}
_NOTE_ALIASES = {"note", "notes"}
_TASK_ALIASES = {"task", "summary", "description"}
_CONTEXT_ALIASES = {"context"}
_MODEL_ALIASES = {"model"}
_CACHE_ALIASES = {"cache"}

_KNOWN_ALIASES: dict[str, str] = {}
for canonical, aliases in (
    ("task", _TASK_ALIASES),
    ("context", _CONTEXT_ALIASES),
    ("args", _ARG_ALIASES),
    ("returns", _RETURN_ALIASES),
    ("raises", _RAISE_ALIASES),
    ("examples", _EXAMPLE_ALIASES),
    ("constraints", _CONSTRAINT_ALIASES),
    ("tools", _TOOLS_ALIASES),
    ("format_hints", _FORMAT_ALIASES),
    ("notes", _NOTE_ALIASES),
    ("model", _MODEL_ALIASES),
    ("cache", _CACHE_ALIASES),
):
    for alias in aliases:
        _KNOWN_ALIASES[alias.lower()] = canonical


_HEADER_NO_VALUE_RE = re.compile(r"^([A-Za-z][A-Za-z _-]*?)\s*:\s*$")
_HEADER_WITH_VALUE_RE = re.compile(r"^([A-Za-z][A-Za-z _-]*?)\s*:\s*(.+?)\s*$")
_DIRECTIVE_KEYS = {"model", "cache"}
_SPHINX_FIELD_RE = re.compile(
    r"^[ \t]*:(param|parameter|arg|argument|type|returns?|rtype|raises?|except|"
    r"raise)\b\s*([^:]*)?:\s*(.*)$",
    re.IGNORECASE,
)
_NUMPY_UNDERLINE_RE = re.compile(r"^-{2,}\s*$")
_DOCTEST_LINE_RE = re.compile(r"^\s*>>>\s?(.*)$")
_DOCTEST_CONT_RE = re.compile(r"^\s*\.\.\.\s?(.*)$")


def parse_docstring(fn: Callable[..., Any]) -> DocstringSpec:
    """Parse ``fn``'s docstring into a :class:`DocstringSpec`.

    Auto-detects Google, Sphinx, or NumPy style based on the structure of the
    body after the leading ``Task`` paragraph.
    """

    raw = inspect.getdoc(fn) or ""
    if not raw.strip():
        return DocstringSpec()

    text = textwrap.dedent(raw).strip("\n")
    style = _detect_style(text)

    if style == "sphinx":
        return _parse_sphinx(text, fn)
    if style == "numpy":
        return _parse_numpy(text, fn)
    return _parse_google(text, fn)


def _detect_style(text: str) -> str:
    lines = text.splitlines()
    has_sphinx_field = any(_SPHINX_FIELD_RE.match(line) for line in lines)
    has_numpy_underline = False
    for i in range(1, len(lines)):
        prev = lines[i - 1].strip()
        cur = lines[i].strip()
        if prev and _NUMPY_UNDERLINE_RE.match(cur) and len(cur) >= len(prev) - 2:
            has_numpy_underline = True
            break
    has_google_header = any(_HEADER_NO_VALUE_RE.match(line) for line in lines)

    if has_sphinx_field and not has_google_header:
        return "sphinx"
    if has_numpy_underline and not has_google_header:
        return "numpy"
    return "google"


def _split_sections_google(
    text: str,
) -> tuple[str, list[tuple[str, str, str]], dict[str, str]]:
    """Split a Google-style body.

    Returns ``(preamble_text, sections, directives)`` where ``sections`` is a
    list of ``(canonical_name, raw_header, body_text)`` and ``directives`` is
    a flat ``{key_lower: value}`` mapping for inline ``Model: foo`` /
    ``Cache: on`` lines that may appear at any nesting.
    """

    lines = text.splitlines()
    sections: list[tuple[str, str, str]] = []
    directives: dict[str, str] = {}
    preamble_lines: list[str] = []
    cur_canonical: str | None = None
    cur_raw: str = ""
    cur_body: list[str] = []

    def flush() -> None:
        nonlocal cur_canonical, cur_raw, cur_body
        if cur_canonical is not None:
            sections.append((cur_canonical, cur_raw, "\n".join(cur_body).rstrip()))
        cur_canonical = None
        cur_raw = ""
        cur_body = []

    for line in lines:
        stripped = line.strip()

        # Inline directive (e.g. "Model: foo") at this dedent level.
        m_dir = _HEADER_WITH_VALUE_RE.match(stripped)
        if (
            m_dir
            and m_dir.group(1).strip().lower() in _DIRECTIVE_KEYS
            and not line.startswith((" ", "\t"))
        ):
            directives[m_dir.group(1).strip().lower()] = m_dir.group(2).strip()
            flush()
            continue

        # Sectional header (e.g. "Args:" with no value).
        m_hdr = _HEADER_NO_VALUE_RE.match(line) if not line.startswith((" ", "\t")) else None
        if m_hdr:
            raw_header = m_hdr.group(1).strip()
            canonical = _KNOWN_ALIASES.get(raw_header.lower(), "__extra__")
            flush()
            cur_canonical = canonical
            cur_raw = raw_header
            cur_body = []
            continue

        if cur_canonical is None:
            preamble_lines.append(line)
        else:
            cur_body.append(line)
    flush()

    preamble = "\n".join(preamble_lines).strip()
    return preamble, sections, directives


def _parse_google(text: str, fn: Callable[..., Any]) -> DocstringSpec:
    spec = DocstringSpec()
    preamble, sections, directives = _split_sections_google(text)

    if preamble:
        spec.task = preamble.strip()

    if not sections and not directives:
        spec.task = text.strip()
        return spec

    for canonical, raw_header, body in sections:
        body = textwrap.dedent(body).strip("\n")
        if canonical == "task":
            spec.task = body.strip() or spec.task
        elif canonical == "context":
            spec.context = body.strip() or None
        elif canonical == "args":
            spec.args.update(_parse_kv_list(body))
        elif canonical == "returns":
            spec.returns = body.strip() or None
        elif canonical == "raises":
            spec.raises.extend(_parse_raises(body, fn))
        elif canonical == "examples":
            spec.examples.extend(_parse_examples(body))
        elif canonical == "constraints":
            spec.constraints.extend(_parse_bulleted(body))
        elif canonical == "tools":
            spec.tools_guidance.update(_parse_kv_list(body))
        elif canonical == "format_hints":
            spec.format_hints = body.strip() or None
        elif canonical == "notes":
            spec.notes = body.strip() or None
        else:
            # Unknown section captured verbatim under its raw header.
            spec.extra[raw_header] = body

    if "model" in directives:
        spec.model = directives["model"].strip() or None
    if "cache" in directives:
        spec.cache = _coerce_cache(directives["cache"], fn)

    return spec


def _parse_sphinx(text: str, fn: Callable[..., Any]) -> DocstringSpec:
    spec = DocstringSpec()
    lines = text.splitlines()

    body_lines: list[str] = []
    field_lines: list[tuple[str, str | None, str]] = []
    in_fields = False

    for line in lines:
        m = _SPHINX_FIELD_RE.match(line)
        if m:
            in_fields = True
            key = (m.group(1) or "").lower()
            name = (m.group(2) or "").strip() or None
            value = (m.group(3) or "").strip()
            field_lines.append((key, name, value))
            continue
        if in_fields and line.strip() and field_lines:
            key, name, value = field_lines[-1]
            field_lines[-1] = (key, name, (value + " " + line.strip()).strip())
            continue
        if in_fields and not line.strip():
            continue
        body_lines.append(line)

    spec.task = "\n".join(body_lines).strip()

    for key, name, value in field_lines:
        if key in {"param", "parameter", "arg", "argument"} and name:
            spec.args[name] = value
        elif key in {"return", "returns"}:
            spec.returns = value or spec.returns
        elif key in {"raise", "raises", "except"}:
            exc_name = name or _first_token(value)
            when_text = value if name else _strip_first_token(value)
            spec.raises.append(_resolve_raise(exc_name, when_text, fn))
        elif key in {"type", "rtype"}:
            continue

    return spec


def _parse_numpy(text: str, fn: Callable[..., Any]) -> DocstringSpec:
    spec = DocstringSpec()
    lines = text.splitlines()

    blocks: list[tuple[str, list[str]]] = []
    cur_header: str | None = None
    cur_body: list[str] = []
    preamble_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and _NUMPY_UNDERLINE_RE.match(next_line.strip()):
            if cur_header is None:
                preamble_lines = list(cur_body)
            else:
                blocks.append((cur_header, cur_body))
            cur_header = line.strip()
            cur_body = []
            i += 2
            continue
        if cur_header is None:
            cur_body.append(line)
        else:
            cur_body.append(line)
        i += 1
    if cur_header is None:
        preamble_lines = cur_body
    else:
        blocks.append((cur_header, cur_body))

    spec.task = "\n".join(preamble_lines).strip()

    for raw_header, body_lines in blocks:
        canonical = _KNOWN_ALIASES.get(raw_header.lower(), "__extra__")
        body = textwrap.dedent("\n".join(body_lines)).strip("\n")
        if canonical == "args":
            spec.args.update(_parse_numpy_args(body))
        elif canonical == "returns":
            spec.returns = _strip_numpy_type_lines(body) or None
        elif canonical == "raises":
            spec.raises.extend(_parse_numpy_raises(body, fn))
        elif canonical == "examples":
            spec.examples.extend(_parse_examples(body))
        elif canonical == "notes":
            spec.notes = body.strip() or None
        elif canonical == "constraints":
            spec.constraints.extend(_parse_bulleted(body))
        elif canonical == "tools":
            spec.tools_guidance.update(_parse_kv_list(body))
        elif canonical == "format_hints":
            spec.format_hints = body.strip() or None
        elif canonical == "context":
            spec.context = body.strip() or None
        elif canonical == "task":
            spec.task = body.strip() or spec.task
        elif canonical == "model":
            spec.model = body.strip() or None
        elif canonical == "cache":
            spec.cache = _coerce_cache(body.strip(), fn)
        else:
            spec.extra[raw_header] = body

    return spec


def _parse_kv_list(body: str) -> dict[str, str]:
    """Parse Google-style ``name: description`` lists.

    Each entry is identified by being at the smallest non-empty indentation
    level in the (already-dedented) body. Continuation lines are joined onto
    the active entry.
    """

    if not body.strip():
        return {}
    out: dict[str, str] = {}
    lines = body.splitlines()

    base_indent: int | None = None
    for line in lines:
        if line.strip():
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if base_indent is None or indent < base_indent:
                base_indent = indent
    base_indent = base_indent or 0

    cur_key: str | None = None
    cur_val: list[str] = []

    def flush() -> None:
        if cur_key is not None:
            out[cur_key] = " ".join(s.strip() for s in cur_val if s.strip())

    entry_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
    for line in lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        m = entry_re.match(line.strip())
        if m and indent == base_indent:
            flush()
            cur_key = m.group(1)
            cur_val = [m.group(2)]
        else:
            cur_val.append(line.strip())
    flush()
    return out


def _parse_raises(body: str, fn: Callable[..., Any]) -> list[RaiseSpec]:
    if not body.strip():
        return []
    out: list[RaiseSpec] = []
    lines = body.splitlines()

    base_indent: int | None = None
    for line in lines:
        if line.strip():
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if base_indent is None or indent < base_indent:
                base_indent = indent
    base_indent = base_indent or 0

    cur_name: str | None = None
    cur_when: list[str] = []

    def flush() -> None:
        if cur_name is not None:
            when = " ".join(s.strip() for s in cur_when if s.strip())
            out.append(_resolve_raise(cur_name, when, fn))

    entry_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*:\s*(.*)$")
    for line in lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        m = entry_re.match(line.strip())
        if m and indent == base_indent:
            flush()
            cur_name = m.group(1)
            cur_when = [m.group(2)]
        else:
            cur_when.append(line.strip())
    flush()
    return out


def _parse_numpy_args(body: str) -> dict[str, str]:
    if not body.strip():
        return {}
    out: dict[str, str] = {}
    lines = body.splitlines()
    cur_name: str | None = None
    cur_desc: list[str] = []

    def flush() -> None:
        if cur_name is not None:
            out[cur_name] = " ".join(s.strip() for s in cur_desc if s.strip())

    for line in lines:
        if not line.startswith((" ", "\t")) and line.strip():
            flush()
            head = line.strip()
            name = head.split(":")[0].strip()
            cur_name = name
            cur_desc = []
        else:
            cur_desc.append(line)
    flush()
    return out


def _parse_numpy_raises(body: str, fn: Callable[..., Any]) -> list[RaiseSpec]:
    if not body.strip():
        return []
    out: list[RaiseSpec] = []
    lines = body.splitlines()
    cur_name: str | None = None
    cur_when: list[str] = []

    def flush() -> None:
        if cur_name is not None:
            when = " ".join(s.strip() for s in cur_when if s.strip())
            out.append(_resolve_raise(cur_name, when, fn))

    for line in lines:
        if not line.startswith((" ", "\t")) and line.strip():
            flush()
            cur_name = line.strip().split()[0].rstrip(":")
            cur_when = []
        else:
            cur_when.append(line)
    flush()
    return out


def _strip_numpy_type_lines(body: str) -> str:
    if not body.strip():
        return ""
    lines = body.splitlines()
    desc_lines: list[str] = []
    for line in lines:
        if line.startswith((" ", "\t")):
            desc_lines.append(line.strip())
    if desc_lines:
        return " ".join(desc_lines)
    return body.strip()


def _parse_examples(body: str) -> list[ExampleSpec]:
    if not body.strip():
        return []
    examples: list[ExampleSpec] = []
    lines = body.splitlines()
    i = 0
    prose_buf: list[str] = []

    def flush_prose() -> None:
        joined = "\n".join(prose_buf).strip()
        if joined:
            examples.append(ExampleSpec(prose=joined))
        prose_buf.clear()

    while i < len(lines):
        line = lines[i]
        m = _DOCTEST_LINE_RE.match(line)
        if m:
            flush_prose()
            input_lines = [m.group(1)]
            i += 1
            while i < len(lines):
                cont = _DOCTEST_CONT_RE.match(lines[i])
                if cont is None:
                    break
                input_lines.append(cont.group(1))
                i += 1
            output_lines: list[str] = []
            while i < len(lines):
                nxt = lines[i]
                if _DOCTEST_LINE_RE.match(nxt) or not nxt.strip():
                    break
                output_lines.append(nxt.strip())
                i += 1
            examples.append(
                ExampleSpec(
                    input_repr="\n".join(input_lines).strip(),
                    output_repr=("\n".join(output_lines).strip() or None),
                )
            )
            continue
        prose_buf.append(line)
        i += 1
    flush_prose()
    return examples


def _parse_bulleted(body: str) -> list[str]:
    if not body.strip():
        return []
    items: list[str] = []
    cur: list[str] = []

    def flush() -> None:
        joined = " ".join(s.strip() for s in cur if s.strip())
        if joined:
            items.append(joined)

    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            cur = []
            continue
        if stripped.startswith(("- ", "* ", "• ")):
            flush()
            cur = [stripped[2:].strip()]
        elif re.match(r"^\d+[.)]\s+", stripped):
            flush()
            cur = [re.sub(r"^\d+[.)]\s+", "", stripped)]
        else:
            cur.append(stripped)
    flush()
    return items


def _coerce_cache(raw: str, fn: Callable[..., Any]) -> CacheMode:
    value = raw.strip().strip("\"'").lower()
    if value not in ("on", "off", "replay"):
        raise ConfigurationError(
            f"Invalid Cache value {raw!r} in docstring of "
            f"{getattr(fn, '__qualname__', fn)!r}; "
            "expected one of 'on', 'off', 'replay'."
        )
    return cast(CacheMode, value)


def _resolve_raise(name: str, when: str, fn: Callable[..., Any]) -> RaiseSpec:
    exc_type = _resolve_exception(name, fn)
    return RaiseSpec(name=name, when=when.strip(), exc_type=exc_type)


def _resolve_exception(name: str, fn: Callable[..., Any]) -> type[BaseException]:
    fn_globals = getattr(fn, "__globals__", {})
    if name in fn_globals:
        candidate = fn_globals[name]
    elif hasattr(builtins, name):
        candidate = getattr(builtins, name)
    else:
        raise ConfigurationError(
            f"Unresolvable exception {name!r} in Raises: section of "
            f"{getattr(fn, '__qualname__', fn)!r}; "
            "must be importable in fn.__globals__ or builtins."
        )
    if not (isinstance(candidate, type) and issubclass(candidate, BaseException)):
        raise ConfigurationError(
            f"{name!r} in Raises: of {getattr(fn, '__qualname__', fn)!r} "
            "did not resolve to a BaseException subclass."
        )
    return candidate


def _first_token(text: str) -> str:
    return text.strip().split()[0] if text.strip() else ""


def _strip_first_token(text: str) -> str:
    parts = text.strip().split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""
