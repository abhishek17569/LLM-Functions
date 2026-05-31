"""Tests for llm_functions._repair."""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_functions._provider import OutputValidationError
from llm_functions._repair import repair_loop

_BASIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def _resp(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@pytest.mark.asyncio
async def test_seed_text_parses_no_calls_needed() -> None:
    """When the seed text is already valid JSON, no provider calls happen."""

    calls: list[dict[str, Any]] = []

    async def provider_call(**kwargs: Any) -> Any:
        calls.append(kwargs)
        raise AssertionError("should not be called")

    out = await repair_loop(
        provider_call=provider_call,
        messages=[{"role": "user", "content": "x"}],
        raw_text=json.dumps({"answer": "ok"}),
        output_schema=_BASIC_SCHEMA,
        model="m",
        temperature=0.0,
        timeout=10.0,
        max_repairs=2,
    )
    assert out == {"answer": "ok"}
    assert calls == []


@pytest.mark.asyncio
async def test_first_attempt_repairs() -> None:
    async def provider_call(**kwargs: Any) -> Any:
        return _resp(json.dumps({"answer": "fixed"}))

    out = await repair_loop(
        provider_call=provider_call,
        messages=[{"role": "user", "content": "x"}],
        raw_text="not json",
        output_schema=_BASIC_SCHEMA,
        model="m",
        temperature=0.0,
        timeout=10.0,
        max_repairs=2,
    )
    assert out == {"answer": "fixed"}


@pytest.mark.asyncio
async def test_exhausts_with_output_validation_error() -> None:
    attempts = ["still bad", "still bad 2"]
    idx = {"i": 0}

    async def provider_call(**kwargs: Any) -> Any:
        i = idx["i"]
        idx["i"] = i + 1
        return _resp(attempts[i])

    with pytest.raises(OutputValidationError) as ei:
        await repair_loop(
            provider_call=provider_call,
            messages=[{"role": "user", "content": "x"}],
            raw_text="not json",
            output_schema=_BASIC_SCHEMA,
            model="m",
            temperature=0.0,
            timeout=10.0,
            max_repairs=2,
        )
    assert ei.value.last_raw_response == "still bad 2"


@pytest.mark.asyncio
async def test_seed_error_skips_initial_parse() -> None:
    """If a seed_error is supplied, repair re-prompts immediately."""

    seed = json.JSONDecodeError("expecting", "doc", 0)

    async def provider_call(**kwargs: Any) -> Any:
        # The repair message must include the failing text and the error string.
        msgs = kwargs["messages"]
        last_user = next(m for m in reversed(msgs) if m["role"] == "user")
        assert "failed validation" in last_user["content"]
        return _resp(json.dumps({"answer": "ok"}))

    out = await repair_loop(
        provider_call=provider_call,
        messages=[{"role": "user", "content": "x"}],
        raw_text="garbage",
        output_schema=_BASIC_SCHEMA,
        model="m",
        temperature=0.0,
        timeout=10.0,
        max_repairs=1,
        seed_error=seed,
    )
    assert out == {"answer": "ok"}


@pytest.mark.asyncio
async def test_provider_failure_during_repair_is_terminal() -> None:
    async def provider_call(**kwargs: Any) -> Any:
        raise RuntimeError("boom")

    with pytest.raises(OutputValidationError) as ei:
        await repair_loop(
            provider_call=provider_call,
            messages=[{"role": "user", "content": "x"}],
            raw_text="not json",
            output_schema=_BASIC_SCHEMA,
            model="m",
            temperature=0.0,
            timeout=10.0,
            max_repairs=2,
        )
    assert "provider call" in str(ei.value)


@pytest.mark.asyncio
async def test_repair_with_zero_max_repairs_fails_immediately() -> None:
    async def provider_call(**kwargs: Any) -> Any:
        raise AssertionError("must not be called when max_repairs=0")

    with pytest.raises(OutputValidationError):
        await repair_loop(
            provider_call=provider_call,
            messages=[{"role": "user", "content": "x"}],
            raw_text="not json",
            output_schema=_BASIC_SCHEMA,
            model="m",
            temperature=0.0,
            timeout=10.0,
            max_repairs=0,
        )
