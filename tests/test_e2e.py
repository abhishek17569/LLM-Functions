"""End-to-end tests for llm_functions.

These tests validate the full ``@llm_function`` pipeline against a fake LiteLLM
``completion_fn`` to record JSON fixtures, then re-run with ``cache="replay"``
to confirm the fixtures play back without any network or LLM call.

Fixtures live under ``tests/fixtures/llm_functions/`` and are committed.
"""

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from llm_functions import llm_function
from llm_functions._config import configure, reset_settings
from llm_functions._decorator import _set_provider_factory
from llm_functions._provider import _OUTPUT_TOOL_NAME, Provider

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "llm_functions"


# ---------------------------------------------------------------------------
# Fake-LiteLLM helpers (shared with test_decorator.py).
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


# ---------------------------------------------------------------------------
# Test setup.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _configure_replay_dir(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the framework at the committed fixtures dir for every test."""

    monkeypatch.setattr("litellm.supports_function_calling", lambda model: True)
    reset_settings()
    configure(replay_fixtures_dir=str(FIXTURES_DIR), allow_all_tools=False)
    yield
    reset_settings()
    _set_provider_factory(Provider)


# ---------------------------------------------------------------------------
# Define the example functions at module scope so __llm_function__ test
# collection works the same way users would expect.
# ---------------------------------------------------------------------------


@llm_function(cache="replay", model="openai/gpt-4o-mini")
def is_man(text: str) -> bool:
    """Decide whether the text is talking about a man.

    Args:
        text: A short sentence about a person.

    Returns:
        True if the subject is a man, False otherwise.
    """


class NameNotFoundError(Exception):
    """No name was found in the text."""


@llm_function(cache="replay", model="openai/gpt-4o-mini")
def find_names(text: str) -> list[str]:
    """Extract every personal name from the text.

    Args:
        text: A passage that may contain names.

    Returns:
        Names in order of first appearance.

    Raises:
        NameNotFoundError: If the text contains no names at all.
    """


@llm_function(cache="replay", model="openai/gpt-4o-mini")
async def greet(name: str) -> AsyncIterator[str]:
    """Stream a friendly greeting.

    Args:
        name: The recipient's name.

    Returns:
        Sentence fragments that concatenate to the full greeting.
    """


# ---------------------------------------------------------------------------
# E2E tests — replay-mode against committed fixtures.
# ---------------------------------------------------------------------------


def test_is_man_returns_true_in_replay_mode() -> None:
    assert is_man("a fellow named John walked in") is True


def test_find_names_returns_list_in_replay_mode() -> None:
    assert find_names("Alice and Bob went home.") == ["Alice", "Bob"]


def test_find_names_raises_declared_exception() -> None:
    """Raise paths are cache-incompatible (an exception is not a value), so we
    drive this test with a fake provider that emits the synthesised
    ``raise_NameNotFoundError`` tool call directly."""

    queue = [
        _resp(
            _msg(
                tool_calls=[
                    _tool_call(
                        call_id="r1",
                        name="raise_NameNotFoundError",
                        arguments={"reason": "no names found in input"},
                    )
                ]
            )
        )
    ]
    fake = FakeCompletion(queue)
    _set_provider_factory(lambda: Provider(completion_fn=fake))

    @llm_function(cache="off", model="openai/gpt-4o-mini")
    def find_names_inline(text: str) -> list[str]:
        """Extract every personal name from the text.

        Args:
            text: A passage that may contain names.

        Returns:
            Names in order of first appearance.

        Raises:
            NameNotFoundError: If the text contains no names at all.
        """

    with pytest.raises(NameNotFoundError, match="no names"):
        find_names_inline("nothing of note happened today")


@pytest.mark.asyncio
async def test_streaming_greet_yields_pieces() -> None:
    pieces: list[str] = []
    async for piece in await greet("Alice"):
        pieces.append(piece)
    assert "Alice" in "".join(pieces)
    assert "".join(pieces).strip() != ""


# ---------------------------------------------------------------------------
# Tools="*" guardrail.
# ---------------------------------------------------------------------------


def test_tools_star_requires_allow_all_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    @llm_function(cache="off", tools="*")
    def fn(text: str) -> str:
        """Echo."""

    # No allow_all_tools=True configured, so calling fn raises.
    from llm_functions._exceptions import ConfigurationError

    # Install a never-called provider so we can run the wrapper at all.
    _set_provider_factory(lambda: Provider(completion_fn=FakeCompletion([])))

    with pytest.raises(ConfigurationError, match="allow_all_tools"):
        fn("hi")


def test_tools_star_works_when_allowed() -> None:
    """When allow_all_tools is True, tools='*' resolves successfully and the
    LLM call is dispatched (returning the canned answer)."""

    configure(allow_all_tools=True)

    queue = [
        _resp(
            _msg(
                tool_calls=[
                    _tool_call(
                        call_id="c1",
                        name=_OUTPUT_TOOL_NAME,
                        arguments={"value": "yes"},
                    )
                ]
            )
        )
    ]
    fake = FakeCompletion(queue)
    _set_provider_factory(lambda: Provider(completion_fn=fake))

    @llm_function(cache="off", tools="*")
    def fn(text: str) -> str:
        """Echo."""

    assert fn("hi") == "yes"


# ---------------------------------------------------------------------------
# Tool-calling end-to-end.
# ---------------------------------------------------------------------------


def test_tool_calling_loop_invokes_user_tool() -> None:
    def search_web(query: str) -> str:
        """Search the web for ``query``."""
        return f"results for {query}"

    queue = [
        _resp(
            _msg(
                tool_calls=[
                    _tool_call(
                        call_id="t1",
                        name="search_web",
                        arguments={"query": "weather paris"},
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
                        arguments={"value": "Paris is rainy."},
                    )
                ]
            )
        ),
    ]
    fake = FakeCompletion(queue)
    _set_provider_factory(lambda: Provider(completion_fn=fake))

    @llm_function(cache="off", tools=[search_web])
    def weather(city: str) -> str:
        """Look up the weather."""

    assert weather("Paris") == "Paris is rainy."
    # Two calls: initial + after tool result.
    assert len(fake.calls) == 2
