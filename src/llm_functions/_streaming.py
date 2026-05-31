"""Streaming dispatchers for llm_function.

Three return shapes layer on top of the raw text-delta stream produced by
:meth:`Provider.complete_stream`:

* :func:`stream_str` — passes deltas through unchanged.
* :func:`stream_iterable` — incrementally parses a top-level JSON array,
  yielding one validated item at a time as it closes.
* :func:`stream_partial` — yields progressively-richer Pydantic instances
  built with :meth:`BaseModel.model_construct` (validation deferred to the
  final emission).

The iterable parser is hand-rolled — :mod:`ijson` is optional and not
present in this project's lock file. The state machine here is intentionally
small (string / escape / depth) and only handles top-level *arrays*.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, TypeVar, cast

from pydantic import BaseModel, TypeAdapter

__all__ = [
    "stream_iterable",
    "stream_partial",
    "stream_str",
]


T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


async def stream_str(provider_stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield text deltas from ``provider_stream`` unchanged.

    Exists as a typed entry point so :class:`forge-decorator <llm_function>`
    can dispatch on ``return_kind="str"`` without special-casing.
    """

    async for delta in provider_stream:
        if delta:
            yield delta


async def stream_iterable(
    provider_stream: AsyncIterator[str],
    item_type: type[T],
) -> AsyncIterator[T]:
    """Yield items from a JSON array as they close.

    The model is expected to emit a top-level JSON array (``[…]``). Each
    fully-formed element is parsed as JSON, validated against ``item_type``
    via :class:`pydantic.TypeAdapter`, and yielded.

    Whitespace, top-level commas, and the surrounding brackets are skipped.
    Any parse or validation failure terminates the stream by raising — the
    caller decides how to recover.
    """

    adapter: TypeAdapter[T] = TypeAdapter(item_type)
    parser = _ArrayItemBuffer()

    async for delta in provider_stream:
        if not delta:
            continue
        for fragment in parser.feed(delta):
            obj = json.loads(fragment)
            yield adapter.validate_python(obj)

    # Drain any trailing item the buffer was holding without a separator.
    for fragment in parser.flush():
        obj = json.loads(fragment)
        yield adapter.validate_python(obj)


async def stream_partial(
    provider_stream: AsyncIterator[str],
    model_type: type[M],
) -> AsyncIterator[M]:
    """Yield progressively-richer Pydantic instances of ``model_type``.

    On each delta we attempt to repair the buffered text into something
    JSON-parseable; if that succeeds and parses to a dict, we emit a model
    via :meth:`BaseModel.model_construct` (no validation). Once the stream
    completes we yield one final, fully-validated instance so the caller
    sees a known-good object at the end.
    """

    buffer = ""
    last_partial: dict[str, Any] | None = None

    async for delta in provider_stream:
        if not delta:
            continue
        buffer += delta
        candidate = _try_parse_object(buffer)
        if candidate is not None and candidate != last_partial:
            last_partial = candidate
            yield model_type.model_construct(**candidate)

    # Final, validated emission. If the buffer never parsed into a dict we
    # still try once more — and let validation errors propagate.
    final_obj = _try_parse_object(buffer) if buffer else None
    if final_obj is None:
        # Last-ditch attempt: strict parse so the caller gets a clear error.
        final_obj = json.loads(buffer) if buffer.strip() else {}
        if not isinstance(final_obj, dict):
            raise ValueError(
                f"stream_partial expected a JSON object, got {type(final_obj).__name__}"
            )
    yield model_type.model_validate(final_obj)


# ---------------------------------------------------------------------------
# JSON-array streaming parser.
# ---------------------------------------------------------------------------


class _ArrayItemBuffer:
    """Stateful parser that emits complete top-level array elements.

    The parser tracks bracket/brace depth, string state, and escape state.
    It ignores everything outside the outermost array, and within the array
    it slices out each element between commas where ``depth == 0`` relative
    to the array.
    """

    def __init__(self) -> None:
        self._depth: int = 0  # nesting depth *inside* the array
        self._in_string: bool = False
        self._escape: bool = False
        self._array_started: bool = False
        self._array_finished: bool = False
        self._current: list[str] = []

    def feed(self, chunk: str) -> list[str]:
        completed: list[str] = []
        for ch in chunk:
            if self._array_finished:
                # Any trailing characters after the closing bracket are
                # ignored — typically just whitespace.
                continue

            if not self._array_started:
                if ch == "[":
                    self._array_started = True
                # Anything before the opening bracket (whitespace, prose) is
                # discarded; pre-array text is not part of the items.
                continue

            if self._in_string:
                self._current.append(ch)
                if self._escape:
                    self._escape = False
                elif ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._in_string = False
                continue

            if ch == '"':
                self._in_string = True
                self._current.append(ch)
                continue

            if ch in "{[":
                self._depth += 1
                self._current.append(ch)
                continue

            if ch == "}":
                self._depth -= 1
                self._current.append(ch)
                continue

            if ch == "]":
                if self._depth == 0:
                    # Closing the outer array — flush any buffered item.
                    item = "".join(self._current).strip()
                    if item:
                        completed.append(item)
                    self._current = []
                    self._array_finished = True
                    continue
                self._depth -= 1
                self._current.append(ch)
                continue

            if ch == "," and self._depth == 0:
                item = "".join(self._current).strip()
                self._current = []
                if item:
                    completed.append(item)
                continue

            self._current.append(ch)

        return completed

    def flush(self) -> list[str]:
        if self._array_finished:
            return []
        item = "".join(self._current).strip()
        self._current = []
        return [item] if item else []


# ---------------------------------------------------------------------------
# Partial-object best-effort parsing.
# ---------------------------------------------------------------------------


def _try_parse_object(text: str) -> dict[str, Any] | None:
    """Best-effort parse of partial JSON text into a dict.

    Strategy: locate the first ``{``, then attempt :func:`json.loads` on the
    substring closing each successive ``}``. Returns the *last* successful
    parse so callers see the richest version available given the buffer.
    Returns ``None`` if no complete object boundary parses cleanly.
    """

    start = text.find("{")
    if start == -1:
        return None

    best: dict[str, Any] | None = None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                # Try to parse text[start:i+1].
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    best = cast(dict[str, Any], parsed)
    return best
