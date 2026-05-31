"""Tests for llmfunctionkit._provider.

All LLM calls are mocked by injecting a fake ``completion_fn`` into
:class:`Provider`; no real network is touched.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from litellm.exceptions import UnsupportedParamsError

from llmfunctionkit._provider import (
    _OUTPUT_TOOL_NAME,
    OutputValidationError,
    Provider,
    ProviderError,
    ToolIterationError,
)

# ---------------------------------------------------------------------------
# Test helpers — build the Pydantic-like response objects LiteLLM returns.
# ---------------------------------------------------------------------------


def _msg(*, content: str | None = None, tool_calls: list[dict[str, Any]] | None = None) -> Any:
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


def _resp(message: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"message": message}]}


def _tool_call(*, call_id: str, name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
    args_text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args_text},
    }


class FakeCompletion:
    """Records calls and replays a queue of canned responses or exceptions.

    Each entry can be an awaitable result *factory* (``Callable[[kwargs], Any]``
    or a literal value/exception). Tests can inspect ``calls`` after to assert
    on what kwargs the provider sent.
    """

    def __init__(self, queue: list[Any]) -> None:
        self._queue = list(queue)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._queue:
            raise AssertionError(
                f"FakeCompletion queue exhausted; unexpected call with kwargs={list(kwargs)}"
            )
        nxt = self._queue.pop(0)
        if callable(nxt) and not isinstance(nxt, BaseException):
            nxt = nxt(kwargs)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


_BASIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _make_provider(queue: list[Any]) -> tuple[Provider, FakeCompletion]:
    fake = FakeCompletion(queue)
    return Provider(completion_fn=fake), fake


# ---------------------------------------------------------------------------
# Native function-calling path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_native_fc_success_emits_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("litellm.supports_function_calling", lambda model: True)
    queue = [
        _resp(
            _msg(
                tool_calls=[
                    _tool_call(call_id="c1", name=_OUTPUT_TOOL_NAME, arguments={"answer": "hi"})
                ]
            )
        )
    ]
    provider, fake = _make_provider(queue)

    result = await provider.complete(
        messages=[{"role": "user", "content": "hi?"}],
        output_schema=_BASIC_SCHEMA,
        model="openai/gpt-4o-mini",
    )
    assert result == {"answer": "hi"}
    assert fake.calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": _OUTPUT_TOOL_NAME},
    }
    # The output-emitter tool must always be present in the request.
    tool_names = [t["function"]["name"] for t in fake.calls[0]["tools"]]
    assert _OUTPUT_TOOL_NAME in tool_names


@pytest.mark.asyncio
async def test_native_fc_unsupported_falls_back_to_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm.supports_function_calling", lambda model: False)
    queue = [_resp(_msg(content=json.dumps({"answer": "json-mode"})))]
    provider, fake = _make_provider(queue)

    result = await provider.complete(
        messages=[{"role": "user", "content": "hi"}],
        output_schema=_BASIC_SCHEMA,
        model="some/unknown-model",
    )
    assert result == {"answer": "json-mode"}
    # JSON-mode path supplies response_format; not tools.
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    assert "tools" not in fake.calls[0]


@pytest.mark.asyncio
async def test_native_fc_unsupported_params_error_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even when supports_function_calling returns True, the provider may
    reject the tool params at request time. Provider must catch and fall back."""

    monkeypatch.setattr("litellm.supports_function_calling", lambda model: True)
    queue = [
        UnsupportedParamsError(
            status_code=400, message="tools not supported here", model="m", llm_provider="x"
        ),
        _resp(_msg(content=json.dumps({"answer": "fallback"}))),
    ]
    provider, fake = _make_provider(queue)

    result = await provider.complete(
        messages=[{"role": "user", "content": "x"}],
        output_schema=_BASIC_SCHEMA,
        model="openai/gpt-4o-mini",
    )
    assert result == {"answer": "fallback"}
    assert len(fake.calls) == 2
    assert "tools" in fake.calls[0]
    assert "tools" not in fake.calls[1]


# ---------------------------------------------------------------------------
# JSON-mode + repair.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_mode_malformed_then_repair_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm.supports_function_calling", lambda model: False)
    queue = [
        _resp(_msg(content="this is not json")),
        _resp(_msg(content=json.dumps({"answer": "fixed"}))),
    ]
    provider, _fake = _make_provider(queue)

    result = await provider.complete(
        messages=[{"role": "user", "content": "x"}],
        output_schema=_BASIC_SCHEMA,
        model="openai/gpt-4o-mini",
        max_repairs=2,
    )
    assert result == {"answer": "fixed"}


@pytest.mark.asyncio
async def test_json_mode_repair_exhausts_raises_output_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm.supports_function_calling", lambda model: False)
    queue = [
        _resp(_msg(content="garbage 1")),
        _resp(_msg(content="garbage 2")),
        _resp(_msg(content="garbage 3")),
    ]
    provider, _fake = _make_provider(queue)

    with pytest.raises(OutputValidationError) as ei:
        await provider.complete(
            messages=[{"role": "user", "content": "x"}],
            output_schema=_BASIC_SCHEMA,
            model="openai/gpt-4o-mini",
            max_repairs=2,
        )
    assert ei.value.last_raw_response == "garbage 3"
    assert isinstance(ei.value.last_error, BaseException)


@pytest.mark.asyncio
async def test_json_mode_non_object_payload_repairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON that parses but isn't an object must trigger repair."""

    monkeypatch.setattr("litellm.supports_function_calling", lambda model: False)
    queue = [
        _resp(_msg(content="[1, 2, 3]")),  # valid JSON, not an object
        _resp(_msg(content=json.dumps({"answer": "now-object"}))),
    ]
    provider, _fake = _make_provider(queue)

    result = await provider.complete(
        messages=[{"role": "user", "content": "x"}],
        output_schema=_BASIC_SCHEMA,
        model="openai/gpt-4o-mini",
        max_repairs=1,
    )
    assert result == {"answer": "now-object"}


@pytest.mark.asyncio
async def test_native_fc_returns_text_falls_through_to_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the model uses 'auto' tool_choice and replies with text, parse it."""

    monkeypatch.setattr("litellm.supports_function_calling", lambda model: True)

    async def _user_tool(name: str, args: dict[str, Any]) -> Any:
        raise AssertionError("user tool should not be called in this test")

    # Provide a user tool so tool_choice becomes "auto".
    queue = [_resp(_msg(content=json.dumps({"answer": "as-text"})))]
    provider, fake = _make_provider(queue)

    result = await provider.complete(
        messages=[{"role": "user", "content": "x"}],
        output_schema=_BASIC_SCHEMA,
        model="openai/gpt-4o-mini",
        tools=[
            {
                "type": "function",
                "function": {"name": "demo_tool", "parameters": {"type": "object"}},
            }
        ],
        tool_invoker=_user_tool,
    )
    assert result == {"answer": "as-text"}
    assert fake.calls[0]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_native_fc_tool_args_malformed_runs_repair_with_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm.supports_function_calling", lambda model: True)
    queue = [
        _resp(
            _msg(
                tool_calls=[
                    _tool_call(
                        call_id="c1",
                        name=_OUTPUT_TOOL_NAME,
                        arguments="{not json",  # raw, malformed
                    )
                ]
            )
        ),
        _resp(_msg(content=json.dumps({"answer": "repaired"}))),
    ]
    provider, _fake = _make_provider(queue)

    result = await provider.complete(
        messages=[{"role": "user", "content": "x"}],
        output_schema=_BASIC_SCHEMA,
        model="openai/gpt-4o-mini",
        max_repairs=1,
    )
    assert result == {"answer": "repaired"}


# ---------------------------------------------------------------------------
# Tool-call dispatch loop.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_dispatch_loop_invokes_tool_then_emits_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm.supports_function_calling", lambda model: True)
    invocations: list[tuple[str, dict[str, Any]]] = []

    async def invoker(name: str, args: dict[str, Any]) -> Any:
        invocations.append((name, args))
        return {"weather": "sunny"}

    queue = [
        _resp(
            _msg(
                tool_calls=[
                    _tool_call(call_id="t1", name="get_weather", arguments={"city": "Paris"})
                ]
            )
        ),
        _resp(
            _msg(
                tool_calls=[
                    _tool_call(
                        call_id="t2",
                        name=_OUTPUT_TOOL_NAME,
                        arguments={"answer": "It's sunny in Paris."},
                    )
                ]
            )
        ),
    ]
    provider, fake = _make_provider(queue)

    result = await provider.complete(
        messages=[{"role": "user", "content": "weather?"}],
        output_schema=_BASIC_SCHEMA,
        model="openai/gpt-4o-mini",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_invoker=invoker,
        max_tool_iterations=3,
    )
    assert result == {"answer": "It's sunny in Paris."}
    assert invocations == [("get_weather", {"city": "Paris"})]
    # The second call must include the assistant tool-call message + tool result.
    second_msgs = fake.calls[1]["messages"]
    roles = [m["role"] for m in second_msgs]
    assert "tool" in roles
    tool_msg = next(m for m in second_msgs if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "t1"
    assert tool_msg["name"] == "get_weather"


@pytest.mark.asyncio
async def test_tool_iteration_overflow_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm.supports_function_calling", lambda model: True)

    async def invoker(name: str, args: dict[str, Any]) -> Any:
        return "ok"

    # Three rounds of user-tool calls; max_tool_iterations=2 trips on the 3rd.
    looping_call = _resp(_msg(tool_calls=[_tool_call(call_id="x", name="loop_tool", arguments={})]))
    queue = [looping_call, looping_call, looping_call]
    provider, _fake = _make_provider(queue)

    with pytest.raises(ToolIterationError) as ei:
        await provider.complete(
            messages=[{"role": "user", "content": "go"}],
            output_schema=_BASIC_SCHEMA,
            model="openai/gpt-4o-mini",
            tools=[
                {
                    "type": "function",
                    "function": {"name": "loop_tool", "parameters": {"type": "object"}},
                }
            ],
            tool_invoker=invoker,
            max_tool_iterations=2,
        )
    assert ei.value.iterations == 3


@pytest.mark.asyncio
async def test_tool_call_without_invoker_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm.supports_function_calling", lambda model: True)
    queue = [_resp(_msg(tool_calls=[_tool_call(call_id="t1", name="some_tool", arguments={})]))]
    provider, _fake = _make_provider(queue)
    with pytest.raises(ProviderError, match="no tool_invoker"):
        await provider.complete(
            messages=[{"role": "user", "content": "x"}],
            output_schema=_BASIC_SCHEMA,
            model="openai/gpt-4o-mini",
            tools=[
                {
                    "type": "function",
                    "function": {"name": "some_tool", "parameters": {"type": "object"}},
                }
            ],
            tool_invoker=None,
        )


@pytest.mark.asyncio
async def test_tool_call_invoker_exception_wraps_as_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm.supports_function_calling", lambda model: True)

    async def bad(name: str, args: dict[str, Any]) -> Any:
        raise RuntimeError("kaboom")

    queue = [_resp(_msg(tool_calls=[_tool_call(call_id="t1", name="bad_tool", arguments={})]))]
    provider, _fake = _make_provider(queue)
    with pytest.raises(ProviderError, match=r"bad_tool.*kaboom"):
        await provider.complete(
            messages=[{"role": "user", "content": "x"}],
            output_schema=_BASIC_SCHEMA,
            model="openai/gpt-4o-mini",
            tools=[
                {
                    "type": "function",
                    "function": {"name": "bad_tool", "parameters": {"type": "object"}},
                }
            ],
            tool_invoker=bad,
        )


@pytest.mark.asyncio
async def test_provider_call_failure_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("litellm.supports_function_calling", lambda model: True)
    queue = [RuntimeError("network down")]
    provider, _fake = _make_provider(queue)
    with pytest.raises(ProviderError, match="completion failed"):
        await provider.complete(
            messages=[{"role": "user", "content": "x"}],
            output_schema=_BASIC_SCHEMA,
            model="openai/gpt-4o-mini",
        )


@pytest.mark.asyncio
async def test_supports_fc_check_failure_treated_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If supports_function_calling raises, fall back to JSON mode."""

    def boom(model: str) -> bool:
        raise RuntimeError("model registry down")

    monkeypatch.setattr("litellm.supports_function_calling", boom)
    queue = [_resp(_msg(content=json.dumps({"answer": "ok"})))]
    provider, _fake = _make_provider(queue)
    result = await provider.complete(
        messages=[{"role": "user", "content": "x"}],
        output_schema=_BASIC_SCHEMA,
        model="weird/model",
    )
    assert result == {"answer": "ok"}


@pytest.mark.asyncio
async def test_json_mode_handles_unsupported_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some providers reject response_format too — provider should retry without."""

    monkeypatch.setattr("litellm.supports_function_calling", lambda model: False)

    def step(kwargs: dict[str, Any]) -> Any:
        if "response_format" in kwargs:
            raise UnsupportedParamsError(
                status_code=400,
                message="response_format not supported",
                model="m",
                llm_provider="x",
            )
        return _resp(_msg(content=json.dumps({"answer": "no-rf"})))

    queue: list[Any] = [step, step]
    provider, fake = _make_provider(queue)
    result = await provider.complete(
        messages=[{"role": "user", "content": "x"}],
        output_schema=_BASIC_SCHEMA,
        model="weird/model",
    )
    assert result == {"answer": "no-rf"}
    assert "response_format" in fake.calls[0]
    assert "response_format" not in fake.calls[1]


@pytest.mark.asyncio
async def test_json_mode_drops_temperature_when_provider_rejects_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPT-5 via TrueFoundry rejects ``temperature=0.0``. Provider should drop it and retry."""

    monkeypatch.setattr("litellm.supports_function_calling", lambda model: False)

    def step(kwargs: dict[str, Any]) -> Any:
        if "temperature" in kwargs:
            raise UnsupportedParamsError(
                status_code=400,
                message=(
                    "litellm.UnsupportedParamsError: gpt-5 models (including gpt-5-codex) "
                    "don't support temperature=0.0. Only temperature=1 is supported."
                ),
                model="openai/gpt-5",
                llm_provider="openai",
            )
        return _resp(_msg(content=json.dumps({"answer": "ok"})))

    queue: list[Any] = [step, step]
    provider, fake = _make_provider(queue)
    result = await provider.complete(
        messages=[{"role": "user", "content": "x"}],
        output_schema=_BASIC_SCHEMA,
        model="openai/gpt-5",
        temperature=0.0,
    )
    assert result == {"answer": "ok"}
    assert "temperature" in fake.calls[0]
    assert "temperature" not in fake.calls[1]
    # response_format is *not* what was rejected, so it should be preserved.
    assert "response_format" in fake.calls[1]


@pytest.mark.asyncio
async def test_native_fc_drops_temperature_then_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native function-calling path should also drop offending params."""

    monkeypatch.setattr("litellm.supports_function_calling", lambda model: True)

    def step(kwargs: dict[str, Any]) -> Any:
        if "temperature" in kwargs:
            raise UnsupportedParamsError(
                status_code=400,
                message="gpt-5 models don't support temperature=0.0",
                model="openai/gpt-5",
                llm_provider="openai",
            )
        return _resp(
            _msg(
                tool_calls=[
                    _tool_call(
                        call_id="c1",
                        name=_OUTPUT_TOOL_NAME,
                        arguments={"answer": "ok"},
                    )
                ]
            )
        )

    queue: list[Any] = [step, step]
    provider, fake = _make_provider(queue)
    result = await provider.complete(
        messages=[{"role": "user", "content": "x"}],
        output_schema=_BASIC_SCHEMA,
        model="openai/gpt-5",
        temperature=0.0,
    )
    assert result == {"answer": "ok"}
    assert "temperature" in fake.calls[0]
    assert "temperature" not in fake.calls[1]
    assert "tools" in fake.calls[1]


@pytest.mark.asyncio
async def test_unrecognised_unsupported_param_is_propagated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If we can't parse the offending param, surface a clear error rather than retry blindly."""

    monkeypatch.setattr("litellm.supports_function_calling", lambda model: False)

    queue: list[Any] = [
        UnsupportedParamsError(
            status_code=400,
            message="some new bizarre constraint we have never seen",
            model="weird/model",
            llm_provider="weird",
        )
    ]
    provider, _fake = _make_provider(queue)
    with pytest.raises(ProviderError):
        await provider.complete(
            messages=[{"role": "user", "content": "x"}],
            output_schema=_BASIC_SCHEMA,
            model="weird/model",
        )


@pytest.mark.asyncio
async def test_tool_with_non_object_args_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm.supports_function_calling", lambda model: True)

    async def invoker(name: str, args: dict[str, Any]) -> Any:
        return "ok"

    queue = [
        _resp(_msg(tool_calls=[_tool_call(call_id="t1", name="weird_tool", arguments="[1,2,3]")]))
    ]
    provider, _fake = _make_provider(queue)
    with pytest.raises(ProviderError, match="must decode to an object"):
        await provider.complete(
            messages=[{"role": "user", "content": "x"}],
            output_schema=_BASIC_SCHEMA,
            model="openai/gpt-4o-mini",
            tools=[
                {
                    "type": "function",
                    "function": {"name": "weird_tool", "parameters": {"type": "object"}},
                }
            ],
            tool_invoker=invoker,
        )


# ---------------------------------------------------------------------------
# Streaming entry point on Provider.
# ---------------------------------------------------------------------------


def _make_chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


class _AsyncList:
    """Minimal async iterator over a list — mimics LiteLLM's stream object."""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)

    def __aiter__(self) -> _AsyncList:
        return self

    async def __anext__(self) -> Any:
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


@pytest.mark.asyncio
async def test_complete_stream_yields_text_deltas() -> None:
    chunks = [_make_chunk("Hello, "), _make_chunk("world"), _make_chunk("!")]

    async def fake_stream(**kwargs: Any) -> Any:
        assert kwargs["stream"] is True
        return _AsyncList(chunks)

    provider = Provider(completion_fn=fake_stream)
    pieces: list[str] = []
    async for delta in provider.complete_stream(
        messages=[{"role": "user", "content": "hi"}],
        model="openai/gpt-4o-mini",
        return_kind="str",
    ):
        pieces.append(delta)
    assert pieces == ["Hello, ", "world", "!"]
