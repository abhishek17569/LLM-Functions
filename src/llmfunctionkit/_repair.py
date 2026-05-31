"""JSON-mode repair loop.

When a model returns text that should be a JSON object but the bytes don't
parse, or parse but fail Pydantic / JSON-Schema validation, this module
re-prompts the model with the failing payload + the validation error and asks
it to retry. Retries are capped at ``max_repairs`` to bound cost.

Validation in this layer is intentionally narrow: we verify that the payload
is a JSON object. Pydantic-model validation of the resulting dict is the
caller's job — forge-decorator owns the typed return-value construction —
which lets the repair loop stay schema-agnostic.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

from litellm.exceptions import UnsupportedParamsError

from ._provider import (
    OutputValidationError,
    _append_schema_instruction,
    _content_of,
    _first_choice,
    _log_request,
    _message_of,
    _params_to_drop,
)

__all__ = ["repair_loop"]


async def repair_loop(
    *,
    provider_call: Callable[..., Awaitable[Any]],
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
    """Parse ``raw_text`` as a JSON object; re-prompt on failure.

    Returns the parsed dict on success. On exhaustion, raises
    :class:`OutputValidationError` with the last error and last raw response
    attached.
    """

    last_error: BaseException | None = seed_error
    last_raw: Any = raw_text
    working_messages: list[dict[str, Any]] = list(messages)
    working_text: str = raw_text

    # Try the seed text first — only re-prompt if the seed_error already
    # disqualifies it, otherwise attempt a parse.
    if seed_error is None:
        try:
            return _parse_object(working_text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            last_raw = working_text

    for _ in range(max_repairs):
        working_messages = _append_repair_message(
            working_messages,
            failing_text=working_text,
            error=last_error,
            output_schema=output_schema,
        )
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": working_messages,
            "temperature": temperature,
            "timeout": timeout,
            "response_format": {"type": "json_object"},
        }
        if api_key is not None:
            call_kwargs["api_key"] = api_key
        if api_base is not None:
            call_kwargs["api_base"] = api_base
        if extra_headers:
            call_kwargs["extra_headers"] = dict(extra_headers)
        _log_request(call_kwargs)
        try:
            response = await provider_call(**call_kwargs)
        except UnsupportedParamsError as exc:
            # Same param-drop retry as the main provider path so a repair
            # attempt isn't killed by, e.g., GPT-5's temperature constraint.
            to_drop = _params_to_drop(exc) & set(call_kwargs)
            if not to_drop:
                raise OutputValidationError(
                    f"repair attempt failed at provider call: {exc}",
                    last_error=exc,
                    last_raw_response=last_raw,
                ) from exc
            retry_kwargs = {k: v for k, v in call_kwargs.items() if k not in to_drop}
            _log_request(retry_kwargs, retry=True)
            try:
                response = await provider_call(**retry_kwargs)
            except Exception as retry_exc:
                raise OutputValidationError(
                    f"repair attempt failed at provider call: {retry_exc}",
                    last_error=retry_exc,
                    last_raw_response=last_raw,
                ) from retry_exc
        except Exception as exc:
            # Treat provider-side errors as terminal repair failures —
            # there's no JSON to salvage.
            raise OutputValidationError(
                f"repair attempt failed at provider call: {exc}",
                last_error=exc,
                last_raw_response=last_raw,
            ) from exc

        message = _message_of(_first_choice(response))
        working_text = _content_of(message) or ""
        last_raw = working_text
        try:
            return _parse_object(working_text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

    assert last_error is not None  # for mypy
    raise OutputValidationError(
        f"output validation failed after {max_repairs} repair attempt(s): {last_error}",
        last_error=last_error,
        last_raw_response=last_raw,
    )


def _parse_object(text: str) -> dict[str, Any]:
    """Parse ``text`` as a JSON object. Non-objects raise :class:`ValueError`."""

    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object, got {type(obj).__name__}")
    return cast(dict[str, Any], obj)


def _append_repair_message(
    messages: list[dict[str, Any]],
    *,
    failing_text: str,
    error: BaseException | None,
    output_schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the next-round message list, including the failing payload."""

    base = list(messages)
    # Always re-state the schema in the repair prompt — a model that already
    # got it wrong benefits from the explicit reminder.
    base = _append_schema_instruction(base, output_schema)
    base.append({"role": "assistant", "content": failing_text})
    base.append(
        {
            "role": "user",
            "content": (
                "Your previous response failed validation:\n"
                f"{error}\n"
                "Return JSON conforming to the schema. Output only the JSON object, "
                "no prose or code fences."
            ),
        }
    )
    return base
