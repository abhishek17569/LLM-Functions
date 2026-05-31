"""Tests for the @llm_function decorator pipeline.

The decorator's pipeline is exercised end-to-end here using a fake LiteLLM
``completion_fn`` injected via ``_set_provider_factory``. No network calls.

We deliberately omit ``from __future__ import annotations`` so that the
return-type annotations of decorated functions resolve to the real classes
rather than forward-reference strings — TypeAdapter needs the real types.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from pydantic import BaseModel

from llm_functions import llm_function
from llm_functions._config import reset_settings
from llm_functions._decorator import _set_provider_factory
from llm_functions._exceptions import ConfigurationError, LLMFunctionError
from llm_functions._provider import (
    _OUTPUT_TOOL_NAME,
    OutputValidationError,
    Provider,
)

# ---------------------------------------------------------------------------
# Test infrastructure: fake LiteLLM completion + provider factory.
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
    def __init__(self, queue: list[Any]) -> None:
        self._queue = list(queue)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._queue:
            raise AssertionError("FakeCompletion queue exhausted")
        nxt = self._queue.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        if callable(nxt):
            return nxt(kwargs)
        return nxt


@pytest.fixture
def install_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    monkeypatch.setattr("litellm.supports_function_calling", lambda model: True)
    yield


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    reset_settings()
    yield
    reset_settings()
    _set_provider_factory(Provider)


def _install_queue(queue: list[Any]) -> FakeCompletion:
    fake = FakeCompletion(queue)
    _set_provider_factory(lambda: Provider(completion_fn=fake))
    return fake


# ---------------------------------------------------------------------------
# Sync wrapper.
# ---------------------------------------------------------------------------


def test_sync_wrapper_returns_typed_value(install_provider: None) -> None:
    _install_queue(
        [
            _resp(
                _msg(
                    tool_calls=[
                        _tool_call(
                            call_id="c1",
                            name=_OUTPUT_TOOL_NAME,
                            arguments={"value": True},
                        )
                    ]
                )
            )
        ]
    )

    @llm_function(cache="off")
    def is_man(text: str) -> bool:
        """Decide whether the text is about a man."""

    assert is_man("a fellow named John") is True


def test_sync_wrapper_returns_list(install_provider: None) -> None:
    _install_queue(
        [
            _resp(
                _msg(
                    tool_calls=[
                        _tool_call(
                            call_id="c1",
                            name=_OUTPUT_TOOL_NAME,
                            arguments={"value": ["Alice", "Bob"]},
                        )
                    ]
                )
            )
        ]
    )

    @llm_function(cache="off")
    def find_names(text: str) -> list[str]:
        """Pull names."""

    assert find_names("Alice and Bob") == ["Alice", "Bob"]


def test_sync_wrapper_returns_pydantic_model(install_provider: None) -> None:
    class Result(BaseModel):
        ok: bool
        message: str

    _install_queue(
        [
            _resp(
                _msg(
                    tool_calls=[
                        _tool_call(
                            call_id="c1",
                            name=_OUTPUT_TOOL_NAME,
                            arguments={"ok": True, "message": "all good"},
                        )
                    ]
                )
            )
        ]
    )

    @llm_function(cache="off")
    def evaluate(input_text: str) -> Result:
        """Decide."""

    out = evaluate("hi")
    assert isinstance(out, Result)
    assert out.ok is True


def test_sync_wrapper_marks_ai_function_attribute() -> None:
    @llm_function
    def fn(x: int) -> int:
        """Double it."""

    assert getattr(fn, "__llm_function__", False) is True
    assert fn.__name__ == "fn"


def test_sync_wrapper_blocks_inside_running_loop(install_provider: None) -> None:
    @llm_function(cache="off")
    def fn(x: int) -> int:
        """Inc."""

    async def driver() -> Any:
        return fn(1)

    with pytest.raises(LLMFunctionError, match="async def"):
        asyncio.run(driver())


# ---------------------------------------------------------------------------
# Async wrapper.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_wrapper_returns_typed_value(install_provider: None) -> None:
    _install_queue(
        [
            _resp(
                _msg(
                    tool_calls=[
                        _tool_call(
                            call_id="c1",
                            name=_OUTPUT_TOOL_NAME,
                            arguments={"value": 42},
                        )
                    ]
                )
            )
        ]
    )

    @llm_function(cache="off")
    async def square(x: int) -> int:
        """Square."""

    assert await square(7) == 42


# ---------------------------------------------------------------------------
# Per-call kwargs override decorator settings.
# ---------------------------------------------------------------------------


def test_per_call_kwargs_override(install_provider: None) -> None:
    fake = _install_queue(
        [
            _resp(
                _msg(
                    tool_calls=[
                        _tool_call(
                            call_id="c1",
                            name=_OUTPUT_TOOL_NAME,
                            arguments={"value": "ok"},
                        )
                    ]
                )
            )
        ]
    )

    @llm_function(cache="off", model="openai/gpt-4o-mini", temperature=0.0)
    def fn(text: str) -> str:
        """Echo."""

    fn("hello", model="anthropic/claude-3.5", temperature=0.7)
    assert fake.calls[0]["model"] == "anthropic/claude-3.5"
    assert fake.calls[0]["temperature"] == 0.7


def _ok_response() -> dict[str, Any]:
    return _resp(
        _msg(
            tool_calls=[
                _tool_call(
                    call_id="c1",
                    name=_OUTPUT_TOOL_NAME,
                    arguments={"value": "ok"},
                )
            ]
        )
    )


def test_configured_provider_forwards_api_key(install_provider: None) -> None:
    from llm_functions import ProviderConfig
    from llm_functions._config import configure

    configure(
        providers={
            "openai": ProviderConfig(api_key="sk-O"),
            "anthropic": ProviderConfig(api_key="sk-A"),
        }
    )

    fake = _install_queue([_ok_response()])

    @llm_function(cache="off", model="openai/gpt-4o-mini")
    def fn(text: str) -> str:
        """Echo."""

    fn("hi")
    assert fake.calls[0]["api_key"] == "sk-O"


def test_provider_picks_anthropic_for_anthropic_model(install_provider: None) -> None:
    from llm_functions import ProviderConfig
    from llm_functions._config import configure

    configure(
        providers={
            "openai": ProviderConfig(api_key="sk-O"),
            "anthropic": ProviderConfig(api_key="sk-A"),
        }
    )

    fake = _install_queue([_ok_response()])

    @llm_function(cache="off", model="anthropic/claude-sonnet-4-6")
    def fn(text: str) -> str:
        """Echo."""

    fn("hi")
    assert fake.calls[0]["api_key"] == "sk-A"


def test_no_credentials_forwarded_when_provider_unconfigured(install_provider: None) -> None:
    from llm_functions import ProviderConfig
    from llm_functions._config import configure

    configure(providers={"openai": ProviderConfig(api_key="sk-O")})

    fake = _install_queue([_ok_response()])

    @llm_function(cache="off", model="anthropic/claude-sonnet-4-6")
    def fn(text: str) -> str:
        """Echo."""

    fn("hi")
    assert "api_key" not in fake.calls[0]
    assert "api_base" not in fake.calls[0]
    assert "extra_headers" not in fake.calls[0]


def test_provider_forwards_api_base_and_extra_headers(install_provider: None) -> None:
    from llm_functions import ProviderConfig
    from llm_functions._config import configure

    configure(
        providers={
            "openai": ProviderConfig(
                api_key="jwt-token",
                api_base="https://llm-gateway.truefoundry.com/api/inference/openai",
                extra_headers={"X-TFY-METADATA": "tenant=foo"},
            ),
        }
    )

    fake = _install_queue([_ok_response()])

    @llm_function(cache="off", model="openai/gpt-4o-mini")
    def fn(text: str) -> str:
        """Echo."""

    fn("hi")
    assert fake.calls[0]["api_key"] == "jwt-token"
    assert fake.calls[0]["api_base"] == "https://llm-gateway.truefoundry.com/api/inference/openai"
    assert fake.calls[0]["extra_headers"] == {"X-TFY-METADATA": "tenant=foo"}


def test_provider_callable_api_key_refreshes_per_call(install_provider: None) -> None:
    from llm_functions import ProviderConfig
    from llm_functions._config import configure

    counter = {"n": 0}

    def fresh_jwt() -> str:
        counter["n"] += 1
        return f"jwt-{counter['n']}"

    configure(providers={"openai": ProviderConfig(api_key=fresh_jwt)})

    fake = _install_queue([_ok_response(), _ok_response()])

    @llm_function(cache="off", model="openai/gpt-4o-mini")
    def fn(text: str) -> str:
        """Echo."""

    fn("hi")
    fn("again")
    assert fake.calls[0]["api_key"] == "jwt-1"
    assert fake.calls[1]["api_key"] == "jwt-2"


def test_debug_kwarg_prints_prompt_to_stderr(
    install_provider: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_queue([_ok_response()])

    @llm_function(cache="off", model="openai/gpt-4o-mini")
    def echo(text: str) -> str:
        """Echo the text back.

        Args:
            text: input string.
        """

    echo("hello world", debug=True)
    err = capsys.readouterr().err
    assert "llm_functions debug:" in err
    assert "echo" in err
    assert "model:       openai/gpt-4o-mini" in err
    assert "--- system ---" in err
    assert "--- user ---" in err
    assert "hello world" in err


def test_debug_off_by_default_emits_no_dump(
    install_provider: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_queue([_ok_response()])

    @llm_function(cache="off", model="openai/gpt-4o-mini")
    def echo(text: str) -> str:
        """Echo."""

    echo("hello")
    err = capsys.readouterr().err
    assert "llm_functions debug" not in err


def test_debug_logging_dumps_request_payload(
    install_provider: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    _install_queue([_ok_response()])

    @llm_function(cache="off", model="openai/gpt-4o-mini")
    def echo(text: str) -> str:
        """Echo."""

    with caplog.at_level(logging.DEBUG, logger="llm_functions"):
        echo("hello")
    debug_records = [
        r for r in caplog.records if r.name == "llm_functions" and r.levelno == logging.DEBUG
    ]
    assert any("acompletion request payload" in r.getMessage() for r in debug_records)
    # Credentials are not present in this call; just verify payload contains messages.
    assert any('"messages"' in r.getMessage() for r in debug_records)


def test_debug_logging_redacts_credentials(
    install_provider: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from llm_functions import ProviderConfig
    from llm_functions._config import configure

    configure(
        providers={
            "openai": ProviderConfig(
                api_key="sk-SECRET-VALUE",
                extra_headers={"X-TFY-METADATA": "tenant=foo"},
            )
        }
    )

    _install_queue([_ok_response()])

    @llm_function(cache="off", model="openai/gpt-4o-mini")
    def echo(text: str) -> str:
        """Echo."""

    with caplog.at_level(logging.DEBUG, logger="llm_functions"):
        echo("hi")
    joined = "\n".join(
        r.getMessage()
        for r in caplog.records
        if r.name == "llm_functions" and r.levelno == logging.DEBUG
    )
    assert "sk-SECRET-VALUE" not in joined
    assert "tenant=foo" not in joined  # extra_headers values redacted
    assert "X-TFY-METADATA" in joined  # but the keys remain visible


def test_arg_mismatch_raises_configuration_error(install_provider: None) -> None:
    @llm_function(cache="off")
    def fn(x: int) -> int:
        """."""

    with pytest.raises(ConfigurationError):
        fn()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Streaming.
# ---------------------------------------------------------------------------


class _AsyncList:
    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)

    def __aiter__(self) -> "_AsyncList":
        return self

    async def __anext__(self) -> Any:
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


@pytest.mark.asyncio
async def test_streaming_str_yields_deltas(install_provider: None) -> None:
    chunks = [_chunk("Hello, "), _chunk("world"), _chunk("!")]

    async def fake(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _AsyncList(chunks)
        raise AssertionError("expected stream")

    _set_provider_factory(lambda: Provider(completion_fn=fake))

    @llm_function(cache="off")
    async def chat(question: str) -> AsyncIterator[str]:
        """Stream the answer."""

    pieces: list[str] = []
    async for piece in await chat("hi"):
        pieces.append(piece)
    assert "".join(pieces) == "Hello, world!"


# ---------------------------------------------------------------------------
# Failure surfaces.
# ---------------------------------------------------------------------------


def test_output_validation_failure_propagates(
    install_provider: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm.supports_function_calling", lambda model: False)
    _install_queue(
        [
            _resp(_msg(content="not json")),
            _resp(_msg(content="still not json")),
            _resp(_msg(content="garbage")),
        ]
    )

    @llm_function(cache="off", max_repairs=2)
    def fn(x: int) -> int:
        """Inc."""

    with pytest.raises(OutputValidationError):
        fn(1)


# ---------------------------------------------------------------------------
# Caching paths.
# ---------------------------------------------------------------------------


def test_cache_on_writes_then_reads(
    install_provider: None,
    tmp_path: Any,
) -> None:
    from llm_functions._config import configure

    configure(cache_dir=str(tmp_path / "cache"))

    fake = _install_queue(
        [
            _resp(
                _msg(
                    tool_calls=[
                        _tool_call(
                            call_id="c1",
                            name=_OUTPUT_TOOL_NAME,
                            arguments={"value": "first"},
                        )
                    ]
                )
            ),
        ]
    )

    @llm_function(cache="on")
    def fn(text: str) -> str:
        """Echo."""

    assert fn("hello") == "first"
    # Second call must come from the cache, not re-invoke the provider.
    assert fn("hello") == "first"
    assert len(fake.calls) == 1


def test_cache_replay_miss_raises(install_provider: None, tmp_path: Any) -> None:
    from llm_functions._config import configure
    from llm_functions._exceptions import ReplayMissError

    configure(replay_fixtures_dir=str(tmp_path / "replay"))

    @llm_function(cache="replay")
    def fn(text: str) -> str:
        """Echo."""

    with pytest.raises(ReplayMissError):
        fn("hello")


# ---------------------------------------------------------------------------
# Synthesised raise-tool path.
# ---------------------------------------------------------------------------


class CustomFailure(Exception):
    """Test exception declared in module globals so docstring resolves it."""


def test_raise_tool_signals_user_exception(install_provider: None) -> None:
    _install_queue(
        [
            _resp(
                _msg(
                    tool_calls=[
                        _tool_call(
                            call_id="r1",
                            name="raise_CustomFailure",
                            arguments={"reason": "no good answer"},
                        )
                    ]
                )
            )
        ]
    )

    @llm_function(cache="off")
    def fn(text: str) -> str:
        """Compute.

        Args:
            text: input.

        Returns:
            The output.

        Raises:
            CustomFailure: When the input is unanswerable.
        """

    with pytest.raises(CustomFailure, match="no good answer"):
        fn("???")


# ---------------------------------------------------------------------------
# Tool dispatch from the decorator.
# ---------------------------------------------------------------------------


def test_decorator_dispatches_user_tool(install_provider: None) -> None:
    invocations: list[tuple[str, Any]] = []

    def lookup(key: str) -> str:
        """Lookup."""
        invocations.append(("lookup", key))
        return f"value-of-{key}"

    queue = [
        _resp(
            _msg(
                tool_calls=[
                    _tool_call(
                        call_id="t1",
                        name="lookup",
                        arguments={"key": "ABC"},
                    )
                ]
            )
        ),
        _resp(
            _msg(
                tool_calls=[
                    _tool_call(
                        call_id="c1",
                        name=_OUTPUT_TOOL_NAME,
                        arguments={"value": "looked up: value-of-ABC"},
                    )
                ]
            )
        ),
    ]
    _install_queue(queue)

    @llm_function(cache="off", tools=[lookup])
    def fn(query: str) -> str:
        """Use lookup."""

    assert fn("anything") == "looked up: value-of-ABC"
    assert invocations == [("lookup", "ABC")]


# ---------------------------------------------------------------------------
# Streaming list of items.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_iterable_int(install_provider: None) -> None:
    chunks = [_chunk("[1,"), _chunk(" 2,"), _chunk(" 3]")]

    async def fake(**kwargs: Any) -> Any:
        return _AsyncList(chunks)

    _set_provider_factory(lambda: Provider(completion_fn=fake))

    @llm_function(cache="off")
    async def numbers() -> AsyncIterator[int]:
        """Stream small integers."""

    seen: list[int] = []
    async for item in await numbers():
        seen.append(item)
    assert seen == [1, 2, 3]


# ---------------------------------------------------------------------------
# Streaming partial Pydantic.
# ---------------------------------------------------------------------------


class StreamModel(BaseModel):
    name: str
    age: int


@pytest.mark.asyncio
async def test_streaming_partial_pydantic(install_provider: None) -> None:
    chunks = [
        _chunk('{"name": "A'),
        _chunk('lice", "age"'),
        _chunk(": 30}"),
    ]

    async def fake(**kwargs: Any) -> Any:
        return _AsyncList(chunks)

    _set_provider_factory(lambda: Provider(completion_fn=fake))

    @llm_function(cache="off")
    async def person() -> AsyncIterator[StreamModel]:
        """Stream a person record."""

    last: StreamModel | None = None
    async for item in await person():
        last = item
    assert last is not None
    assert last.name == "Alice"
    assert last.age == 30
