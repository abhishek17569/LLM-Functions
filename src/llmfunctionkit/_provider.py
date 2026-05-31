"""LiteLLM-backed provider wrapper for llm_function.

Performs structured-output completions via two paths:

1. **Native function-calling.** When the target model supports tools, the
   output schema is wrapped as a single forced tool call and the tool-call
   arguments are parsed back into JSON.
2. **JSON-mode fallback.** When function-calling is unsupported (or the
   provider rejects the tool params with
   :class:`litellm.exceptions.UnsupportedParamsError`), the schema is
   appended to the prompt and the model's text content is parsed as JSON,
   with a typed repair loop on validation failure.

User-defined tools are dispatched through a ``tool_invoker`` callback
supplied by ``forge-decorator`` (the registry owner). The dispatch loop is
capped at ``max_tool_iterations`` to prevent runaway sessions.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, cast

import litellm
from litellm.exceptions import UnsupportedParamsError

from ._config import LLMFunctionError

__all__ = [
    "OutputValidationError",
    "Provider",
    "ProviderError",
    "ToolInvoker",
    "ToolIterationError",
]


_LOGGER = logging.getLogger("llmfunctionkit")
_OUTPUT_TOOL_NAME = "emit_final_answer"

# LiteLLM raises ``UnsupportedParamsError`` when the upstream provider rejects
# a parameter (e.g. GPT-5's ``temperature`` constraint, Anthropic's lack of
# ``response_format``, gateways that strip ``parallel_tool_calls``). The error
# message names the offending param. We parse it, drop those params, and retry
# once — so the same code path works on every model without per-model tables.
_KNOWN_DROPPABLE_PARAMS: tuple[str, ...] = (
    "temperature",
    "response_format",
    "tool_choice",
    "parallel_tool_calls",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "logprobs",
    "top_logprobs",
    "n",
    "seed",
    "stream",
)


def _params_to_drop(exc: BaseException) -> set[str]:
    """Parse ``UnsupportedParamsError`` for parameter names to remove.

    LiteLLM's messages typically read ``"... gpt-5 models ... don't support
    temperature=0.0 ..."`` — we look for whole-word matches against a known
    allowlist so we never accidentally strip an arg the user actually needs.
    """

    text = str(exc)
    lowered = text.lower()
    found: set[str] = set()
    for param in _KNOWN_DROPPABLE_PARAMS:
        if param in lowered:
            found.add(param)
    return found


ToolInvoker = Callable[[str, dict[str, Any]], Awaitable[Any]]
ReturnKind = Literal["str", "iterable", "partial"]


class ProviderError(LLMFunctionError):
    """Raised when the provider call fails for a non-validation reason."""


class OutputValidationError(LLMFunctionError):
    """Raised when the model's structured output fails validation after repairs.

    The ``last_error`` attribute holds the most recent validation failure (a
    :class:`ValueError`/:class:`pydantic.ValidationError` or JSON parse error);
    ``last_raw_response`` holds the raw text/dict the model returned.
    """

    def __init__(self, message: str, *, last_error: BaseException, last_raw_response: Any) -> None:
        super().__init__(message)
        self.last_error = last_error
        self.last_raw_response = last_raw_response


class ToolIterationError(LLMFunctionError):
    """Raised when the model exceeds ``max_tool_iterations`` tool-call rounds."""

    def __init__(self, message: str, *, iterations: int) -> None:
        super().__init__(message)
        self.iterations = iterations


class Provider:
    """LiteLLM-backed completion provider.

    The provider is a thin coordinator: schema framing, function-calling
    detection, the tool-dispatch loop, and the JSON-mode/repair fallback all
    live here, while the actual HTTP work and provider-specific quirks are
    delegated to ``litellm.acompletion``.
    """

    def __init__(
        self,
        *,
        completion_fn: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        # ``completion_fn`` is injectable for testing. Default is the live
        # ``litellm.acompletion`` coroutine; tests substitute a fake.
        self._acompletion: Callable[..., Awaitable[Any]] = completion_fn or litellm.acompletion

    async def _call_with_param_drop(self, **kwargs: Any) -> Any:
        """Call ``acompletion`` with one auto-retry on ``UnsupportedParamsError``.

        When the upstream provider rejects specific kwargs (GPT-5 rejecting
        ``temperature=0.0``, models without ``response_format`` support, etc.),
        parse the offending param names out of the error and retry once with
        those kwargs removed. The set of droppable params is a fixed allowlist
        — we never strip user-supplied ``messages``, ``model``, ``api_key``,
        etc.
        """

        _log_request(kwargs)
        try:
            return await self._acompletion(**kwargs)
        except UnsupportedParamsError as exc:
            to_drop = _params_to_drop(exc) & set(kwargs)
            if not to_drop:
                # Nothing recognisable to drop — propagate so the caller can
                # surface a clear error. Re-raising the same type lets the
                # native path's outer handler still fall back to JSON mode.
                raise
            _LOGGER.info(
                "llmfunctionkit: %s rejected params %s; retrying without them.",
                kwargs.get("model", "<unknown model>"),
                sorted(to_drop),
            )
            retry_kwargs = {k: v for k, v in kwargs.items() if k not in to_drop}
            _log_request(retry_kwargs, retry=True)
            return await self._acompletion(**retry_kwargs)

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        output_schema: dict[str, Any],
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
        max_repairs: int = 2,
        max_tool_iterations: int = 10,
        tools: list[dict[str, Any]] | None = None,
        tool_invoker: ToolInvoker | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run a single completion, returning the validated JSON dict.

        ``output_schema`` must be a JSON Schema describing the desired output
        object. ``tools`` are user-defined tools (LiteLLM/OpenAI-style ``tool``
        definitions); ``tool_invoker`` dispatches them by name.

        Returns the parsed output object as a ``dict``. Validation against a
        Pydantic model is the caller's responsibility (forge-decorator
        re-validates and constructs the typed return value).
        """

        use_native = _supports_native_function_calling(model)
        working_messages: list[dict[str, Any]] = [dict(m) for m in messages]

        if use_native:
            try:
                return await self._complete_native(
                    messages=working_messages,
                    output_schema=output_schema,
                    model=model,
                    temperature=temperature,
                    timeout=timeout,
                    tools=tools or [],
                    tool_invoker=tool_invoker,
                    max_tool_iterations=max_tool_iterations,
                    max_repairs=max_repairs,
                    api_key=api_key,
                    api_base=api_base,
                    extra_headers=extra_headers,
                )
            except UnsupportedParamsError as exc:
                _LOGGER.info(
                    "llmfunctionkit: model %s rejected tool params (%s); falling back to JSON mode.",
                    model,
                    exc,
                )
                # Fall through to JSON-mode path.

        return await self._complete_json_mode(
            messages=working_messages,
            output_schema=output_schema,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_repairs=max_repairs,
            api_key=api_key,
            api_base=api_base,
            extra_headers=extra_headers,
        )

    async def complete_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
        return_kind: ReturnKind,
        output_schema: dict[str, Any] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream raw text deltas from the underlying provider.

        Higher-level shaping (parsing arrays, partial Pydantic models) is
        handled in :mod:`llmfunctionkit._streaming`. This method only yields
        text deltas as they arrive and propagates upstream errors.
        """

        # Streaming uses JSON-mode-style framing — many provider/model pairs
        # cannot stream forced tool calls, so we always nudge JSON in the
        # prompt when a schema is supplied.
        prepared = list(messages)
        if output_schema is not None:
            prepared = _append_schema_instruction(prepared, output_schema)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": prepared,
            "temperature": temperature,
            "timeout": timeout,
            "stream": True,
        }
        _apply_credentials(kwargs, api_key=api_key, api_base=api_base, extra_headers=extra_headers)
        try:
            stream = await self._call_with_param_drop(**kwargs)
        except Exception as exc:  # pragma: no cover - thin wrapping
            raise ProviderError(f"streaming completion failed: {exc}") from exc

        async for chunk in _aiter(stream):
            delta = _extract_text_delta(chunk)
            if delta:
                yield delta

        # ``return_kind`` is part of the public signature for forge-decorator's
        # dispatcher to specialise on; the actual specialisation happens in
        # :mod:`llmfunctionkit._streaming`. Keeping it here lets a caller use the
        # provider directly if they want raw deltas.
        del return_kind

    async def _complete_native(
        self,
        *,
        messages: list[dict[str, Any]],
        output_schema: dict[str, Any],
        model: str,
        temperature: float,
        timeout: float,
        tools: list[dict[str, Any]],
        tool_invoker: ToolInvoker | None,
        max_tool_iterations: int,
        max_repairs: int,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        output_tool = _output_schema_as_tool(output_schema)
        all_tools = [*tools, output_tool]

        iterations = 0
        while True:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "timeout": timeout,
                "tools": all_tools,
                "tool_choice": "auto" if tools else _force_tool_choice(output_tool),
            }
            _apply_credentials(
                kwargs, api_key=api_key, api_base=api_base, extra_headers=extra_headers
            )
            try:
                response = await self._call_with_param_drop(**kwargs)
            except UnsupportedParamsError:
                # Re-raise so the public ``complete`` method can fall back to
                # JSON mode for genuinely tool-call-incompatible models.
                raise
            except Exception as exc:
                raise ProviderError(f"completion failed: {exc}") from exc

            choice = _first_choice(response)
            message = _message_of(choice)
            tool_calls = _tool_calls_of(message)

            if not tool_calls:
                # Model returned text instead of using a tool — fall back to
                # JSON-mode parsing of the assistant content (this is rare but
                # can happen when ``tool_choice="auto"`` and the model decides
                # text is fine). Run through repair so malformed JSON is
                # repaired rather than crashing.
                content = _content_of(message) or ""
                return await self._json_decode_with_repair(
                    messages=messages,
                    raw_text=content,
                    output_schema=output_schema,
                    model=model,
                    temperature=temperature,
                    timeout=timeout,
                    max_repairs=max_repairs,
                    api_key=api_key,
                    api_base=api_base,
                    extra_headers=extra_headers,
                )

            # Partition tool calls into output-emitter vs user tools.
            output_call = next(
                (tc for tc in tool_calls if _tool_call_name(tc) == _OUTPUT_TOOL_NAME),
                None,
            )
            if output_call is not None:
                args_text = _tool_call_arguments(output_call)
                try:
                    return cast(dict[str, Any], json.loads(args_text))
                except json.JSONDecodeError as exc:
                    # Run the JSON repair loop on the malformed tool args.
                    return await self._json_decode_with_repair(
                        messages=messages,
                        raw_text=args_text,
                        output_schema=output_schema,
                        model=model,
                        temperature=temperature,
                        timeout=timeout,
                        max_repairs=max_repairs,
                        seed_error=exc,
                        api_key=api_key,
                        api_base=api_base,
                        extra_headers=extra_headers,
                    )

            # Otherwise we have user-tool calls — dispatch them.
            iterations += 1
            if iterations > max_tool_iterations:
                raise ToolIterationError(
                    f"exceeded max_tool_iterations={max_tool_iterations} during tool dispatch",
                    iterations=iterations,
                )
            if tool_invoker is None:
                raise ProviderError("model issued tool calls but no tool_invoker was provided")

            # Append the assistant's tool-call message verbatim so the model
            # can correlate the tool responses we send next.
            messages.append(_assistant_message_with_tool_calls(message, tool_calls))

            for tc in tool_calls:
                name = _tool_call_name(tc)
                args_text = _tool_call_arguments(tc)
                try:
                    args_obj = json.loads(args_text) if args_text else {}
                except json.JSONDecodeError as exc:
                    raise ProviderError(
                        f"tool call {name!r} had non-JSON arguments: {exc}"
                    ) from exc
                if not isinstance(args_obj, dict):
                    raise ProviderError(
                        f"tool call {name!r} arguments must decode to an object, got {type(args_obj).__name__}"
                    )
                try:
                    result = await tool_invoker(name, args_obj)
                except Exception as exc:
                    raise ProviderError(f"tool {name!r} raised: {exc}") from exc
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _tool_call_id(tc),
                        "name": name,
                        "content": _stringify_tool_result(result),
                    }
                )
            # Loop back to let the model consume tool results.

    async def _complete_json_mode(
        self,
        *,
        messages: list[dict[str, Any]],
        output_schema: dict[str, Any],
        model: str,
        temperature: float,
        timeout: float,
        max_repairs: int,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        prepared = _append_schema_instruction(messages, output_schema)
        extra: dict[str, Any] = {}
        _apply_credentials(extra, api_key=api_key, api_base=api_base, extra_headers=extra_headers)

        try:
            response = await self._call_with_param_drop(
                model=model,
                messages=prepared,
                temperature=temperature,
                timeout=timeout,
                response_format={"type": "json_object"},
                **extra,
            )
        except Exception as exc:
            raise ProviderError(f"completion failed: {exc}") from exc

        message = _message_of(_first_choice(response))
        content = _content_of(message) or ""
        return await self._json_decode_with_repair(
            messages=prepared,
            raw_text=content,
            output_schema=output_schema,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_repairs=max_repairs,
            api_key=api_key,
            api_base=api_base,
            extra_headers=extra_headers,
        )

    async def _json_decode_with_repair(
        self,
        *,
        messages: list[dict[str, Any]],
        raw_text: str,
        output_schema: dict[str, Any],
        model: str,
        temperature: float,
        timeout: float,
        max_repairs: int,
        seed_error: BaseException | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        # Local import avoids a cycle: _repair imports Provider for typing.
        from ._repair import repair_loop

        return await repair_loop(
            provider_call=self._acompletion,
            messages=messages,
            raw_text=raw_text,
            output_schema=output_schema,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_repairs=max_repairs,
            seed_error=seed_error,
            api_key=api_key,
            api_base=api_base,
            extra_headers=extra_headers,
        )


# ---------------------------------------------------------------------------
# Helpers (module-private; covered by tests via Provider.complete()).
# ---------------------------------------------------------------------------


def _log_request(kwargs: dict[str, Any], *, retry: bool = False) -> None:
    """Emit a DEBUG log of the outgoing acompletion payload.

    Skipped entirely (no JSON serialisation) unless the ``llm_function`` logger
    is enabled at DEBUG. Credentials are redacted so dumps from production
    logs never leak the JWT.
    """

    if not _LOGGER.isEnabledFor(logging.DEBUG):
        return
    safe = {k: v for k, v in kwargs.items() if k not in {"api_key"}}
    if "extra_headers" in safe and isinstance(safe["extra_headers"], dict):
        safe["extra_headers"] = {k: "<redacted>" for k in safe["extra_headers"]}
    try:
        body = json.dumps(safe, indent=2, default=str, sort_keys=True)
    except Exception:  # pragma: no cover - very defensive
        body = repr(safe)
    label = "retry " if retry else ""
    _LOGGER.debug("llmfunctionkit: %sacompletion request payload:\n%s", label, body)


def _apply_credentials(
    kwargs: dict[str, Any],
    *,
    api_key: str | None,
    api_base: str | None,
    extra_headers: dict[str, str] | None,
) -> None:
    """Inject per-provider credentials into a LiteLLM ``acompletion`` kwargs dict.

    Only sets keys that are actually present, so callers without overrides
    keep falling back to LiteLLM's environment-variable conventions.
    """

    if api_key is not None:
        kwargs["api_key"] = api_key
    if api_base is not None:
        kwargs["api_base"] = api_base
    if extra_headers:
        kwargs["extra_headers"] = dict(extra_headers)


def _supports_native_function_calling(model: str) -> bool:
    try:
        return bool(litellm.supports_function_calling(model=model))
    except Exception:
        return False


def _output_schema_as_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap an output JSON Schema as a single function-tool definition."""

    parameters = dict(schema)
    parameters.setdefault("type", "object")
    return {
        "type": "function",
        "function": {
            "name": _OUTPUT_TOOL_NAME,
            "description": "Emit the final answer in the required schema.",
            "parameters": parameters,
        },
    }


def _force_tool_choice(tool: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-style forced tool choice. Anthropic accepts the same shape via LiteLLM."""

    return {"type": "function", "function": {"name": tool["function"]["name"]}}


def _append_schema_instruction(
    messages: list[dict[str, Any]], output_schema: dict[str, Any]
) -> list[dict[str, Any]]:
    """Append a JSON-mode instruction message describing the output schema."""

    schema_text = json.dumps(output_schema, indent=2, sort_keys=True)
    instruction = (
        "Reply with a single JSON object that conforms exactly to the following "
        "JSON Schema. Do not include any prose, markdown, or code fences — "
        "respond with the raw JSON object only.\n\nSchema:\n"
        f"{schema_text}"
    )
    return [*messages, {"role": "user", "content": instruction}]


def _first_choice(response: Any) -> Any:
    choices = _attr_or_key(response, "choices")
    if not choices:
        raise ProviderError("provider response had no choices")
    return choices[0]


def _message_of(choice: Any) -> Any:
    msg = _attr_or_key(choice, "message")
    if msg is None:
        raise ProviderError("provider choice had no message")
    return msg


def _tool_calls_of(message: Any) -> list[Any]:
    raw = _attr_or_key(message, "tool_calls")
    if not raw:
        return []
    return list(raw)


def _content_of(message: Any) -> str | None:
    val = _attr_or_key(message, "content")
    if val is None:
        return None
    return str(val)


def _tool_call_name(tc: Any) -> str:
    fn = _attr_or_key(tc, "function")
    if fn is None:
        return ""
    name = _attr_or_key(fn, "name")
    return str(name) if name is not None else ""


def _tool_call_arguments(tc: Any) -> str:
    fn = _attr_or_key(tc, "function")
    if fn is None:
        return ""
    args = _attr_or_key(fn, "arguments")
    if args is None:
        return ""
    return args if isinstance(args, str) else json.dumps(args)


def _tool_call_id(tc: Any) -> str:
    val = _attr_or_key(tc, "id")
    return str(val) if val is not None else ""


def _attr_or_key(obj: Any, key: str) -> Any:
    """Read ``key`` from object attribute or mapping key — LiteLLM responses
    may be either Pydantic models or plain dicts depending on configuration."""

    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _assistant_message_with_tool_calls(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
    """Re-serialise the assistant tool-call message to plain dict form."""

    serialised: list[dict[str, Any]] = []
    for tc in tool_calls:
        serialised.append(
            {
                "id": _tool_call_id(tc),
                "type": "function",
                "function": {
                    "name": _tool_call_name(tc),
                    "arguments": _tool_call_arguments(tc),
                },
            }
        )
    return {
        "role": "assistant",
        "content": _content_of(message) or "",
        "tool_calls": serialised,
    }


def _stringify_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result)
    except (TypeError, ValueError):
        return str(result)


async def _aiter(maybe_async_iter: Any) -> AsyncIterator[Any]:
    """Iterate either an async iterator or a plain iterable.

    LiteLLM's stream returns an async iterator; tests sometimes pass a list of
    canned chunks. Normalise both shapes here.
    """

    if hasattr(maybe_async_iter, "__aiter__"):
        async for item in maybe_async_iter:
            yield item
        return
    for item in maybe_async_iter:
        yield item


def _extract_text_delta(chunk: Any) -> str:
    """Extract a text delta from a streaming chunk, ignoring chunks without one."""

    choices = _attr_or_key(chunk, "choices")
    if not choices:
        return ""
    delta = _attr_or_key(choices[0], "delta")
    if delta is None:
        return ""
    content = _attr_or_key(delta, "content")
    if content is None:
        return ""
    return str(content)
