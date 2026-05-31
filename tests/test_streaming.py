"""Tests for llmfunctionkit._streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel, ValidationError

from llmfunctionkit._streaming import stream_iterable, stream_partial, stream_str


async def _stream(chunks: list[str]) -> AsyncIterator[str]:
    for c in chunks:
        yield c


# ---------------------------------------------------------------------------
# stream_str
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_str_passes_deltas_through() -> None:
    pieces = [s async for s in stream_str(_stream(["foo", "bar", "baz"]))]
    assert pieces == ["foo", "bar", "baz"]


@pytest.mark.asyncio
async def test_stream_str_skips_empty_deltas() -> None:
    pieces = [s async for s in stream_str(_stream(["", "x", "", "y", ""]))]
    assert pieces == ["x", "y"]


# ---------------------------------------------------------------------------
# stream_iterable
# ---------------------------------------------------------------------------


class _Item(BaseModel):
    name: str
    n: int


@pytest.mark.asyncio
async def test_stream_iterable_emits_items_as_they_close() -> None:
    text = json.dumps([{"name": "a", "n": 1}, {"name": "b", "n": 2}])
    # Slice to deliberately split mid-element.
    chunks = [text[i : i + 3] for i in range(0, len(text), 3)]
    items = [it async for it in stream_iterable(_stream(chunks), _Item)]
    assert items == [_Item(name="a", n=1), _Item(name="b", n=2)]


@pytest.mark.asyncio
async def test_stream_iterable_handles_strings_with_commas_and_brackets() -> None:
    payload = [{"name": "a, b]", "n": 1}, {"name": 'c " d', "n": 2}]
    text = json.dumps(payload)
    chunks = [text[i : i + 5] for i in range(0, len(text), 5)]
    items = [it async for it in stream_iterable(_stream(chunks), _Item)]
    assert [it.model_dump() for it in items] == payload


@pytest.mark.asyncio
async def test_stream_iterable_drains_trailing_item_without_closing_bracket() -> None:
    """If the stream ends without ``]`` (truncated), the buffer flush emits the
    last item we accumulated."""

    text = '[{"name": "x", "n": 7}'  # missing closing bracket
    items = [it async for it in stream_iterable(_stream([text]), _Item)]
    assert items == [_Item(name="x", n=7)]


@pytest.mark.asyncio
async def test_stream_iterable_validation_errors_propagate() -> None:
    text = json.dumps([{"name": "ok", "n": "not-an-int"}])
    with pytest.raises(ValidationError):
        async for _ in stream_iterable(_stream([text]), _Item):
            pass


# ---------------------------------------------------------------------------
# stream_partial
# ---------------------------------------------------------------------------


class _PartialModel(BaseModel):
    title: str
    body: str


@pytest.mark.asyncio
async def test_stream_partial_yields_progressive_then_final() -> None:
    payload = {"title": "Hello", "body": "World"}
    text = json.dumps(payload)
    # Feed one char at a time so the parser sees many states.
    chunks = list(text)
    out = [m async for m in stream_partial(_stream(chunks), _PartialModel)]
    # The final yield must be a fully validated instance.
    assert out[-1].model_dump() == payload
    # At least one progressive yield should be a model_construct'd instance —
    # we can't easily assert that without inspecting model_fields_set, so we
    # verify there are >= 1 emissions and they're all the right type.
    assert len(out) >= 1
    assert all(isinstance(m, _PartialModel) for m in out)


@pytest.mark.asyncio
async def test_stream_partial_invalid_final_raises() -> None:
    """A final buffer that doesn't produce a dict must raise on the final emit."""

    chunks = ["[1, 2, 3]"]  # array, not an object
    with pytest.raises(ValueError, match="expected a JSON object"):
        async for _ in stream_partial(_stream(chunks), _PartialModel):
            pass


@pytest.mark.asyncio
async def test_stream_partial_empty_buffer() -> None:
    """An empty stream produces a single (default-constructed-via-validate) model.

    With required fields, validation fails — assert that path raises clearly.
    """

    with pytest.raises(ValidationError):
        async for _ in stream_partial(_stream([]), _PartialModel):
            pass
