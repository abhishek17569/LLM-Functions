"""The ``@llm_function`` decorator — wires every llm_function module together.

Pipeline at call time:

1. Resolve runtime settings (defaults < global < decorator < docstring < call).
2. Build the JSON schemas for input and output, plus the synthesised
   ``raise_<Exc>`` tools from any ``Raises:`` directives.
3. Compute the cache key from ``(fn source, model, system prompt, tool
   schemas, args, sampling params)``.
4. Consult the appropriate cache:
   * ``cache="on"`` → :class:`ResultCache` first.
   * ``cache="replay"`` → :class:`ReplayStore`; missing fixture raises
     :class:`ReplayMissError`.
5. Build the messages list (system prompt + user message with arg values).
6. Resolve user tools and prepend the synthesised raise tools.
7. Dispatch to :class:`Provider`:
   * Non-streaming → :meth:`Provider.complete`.
   * Streaming → :meth:`Provider.complete_stream` wrapped in the matching
     :mod:`llm_functions._streaming` shaper.
8. Validate the provider's dict against the return type via
   :class:`pydantic.TypeAdapter`.
9. Cache the assembled value (streaming results are fully materialised first).

Sync vs async dispatch: ``inspect.iscoroutinefunction`` decides which wrapper
shape to return. The sync wrapper detects an active event loop and refuses to
block it; users can switch to the async function in that case.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, cast, get_args, get_origin

from pydantic import BaseModel, TypeAdapter

from ._cache import MISS, ResultCache, make_key
from ._config import Settings, resolve
from ._docstring import DocstringSpec, parse_docstring
from ._exceptions import (
    ConfigurationError,
    LLMFunctionError,
    _LLMFunctionRaisedException,
    synthesise_raise_tools,
)
from ._provider import Provider
from ._replay import ReplayStore
from ._schema import build_input_schema, build_output_schema
from ._streaming import stream_iterable, stream_partial, stream_str
from ._tools import ToolDef, invoke_tool, resolve_tools

__all__ = ["llm_function"]


_PER_CALL_KWARGS: frozenset[str] = frozenset(
    {"cache", "model", "temperature", "max_repairs", "timeout"}
)


_provider_factory: Callable[[], Provider] = Provider


def _set_provider_factory(factory: Callable[[], Provider]) -> None:
    """Override the :class:`Provider` constructor used by the pipeline.

    Tests use this to inject a Provider with a fake ``completion_fn`` without
    monkeypatching litellm.acompletion globally.
    """

    global _provider_factory
    _provider_factory = factory


def _build_cache_key_for_call(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Compute the cache key the wrapper would use for ``fn(*args, **kwargs)``.

    Used by the fixture-recording test helper so it can write replay JSON to
    the right path. The function ``fn`` must already be wrapped by
    :func:`llm_function`.
    """

    target = getattr(fn, "__wrapped__", fn)
    return_kind = _detect_return_kind(target)
    sig = inspect.signature(target)
    decorator_kwargs = cast(
        dict[str, Any], getattr(fn, "__llm_function_decorator_kwargs__", {}) or {}
    )
    user_tools_spec = getattr(fn, "__llm_function_tools_spec__", None)
    ctx = _FunctionContext(
        fn=target,
        sig=sig,
        return_kind=return_kind,
        decorator_kwargs=decorator_kwargs,
        user_tools_spec=user_tools_spec,
    )
    call_kwargs, fn_kwargs = _split_per_call_kwargs(kwargs)
    bound = _bind_args(ctx, args, fn_kwargs)
    arg_dict = dict(bound.arguments)
    settings = resolve(
        decorator=ctx.decorator_kwargs,
        docstring=_docstring_to_settings_layer(ctx.docstring),
        call=call_kwargs,
    )
    user_tools = resolve_tools(ctx.user_tools_spec, settings)
    raise_tool_schemas = synthesise_raise_tools(ctx.docstring.raises)
    system_prompt = _build_system_prompt(ctx)
    tool_schemas = [t.to_openai() for t in user_tools] + raise_tool_schemas
    sampling = {
        "temperature": settings.temperature,
        "max_repairs": settings.max_repairs,
        "max_tool_iterations": settings.max_tool_iterations,
    }
    return make_key(
        fn=target,
        model=settings.model,
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        args=arg_dict,
        kwargs={},
        sampling=sampling,
    )


def llm_function(
    fn: Callable[..., Any] | None = None,
    *,
    tools: list[Callable[..., Any]] | Literal["*"] | None = None,
    model: str | None = None,
    cache: Literal["on", "off", "replay"] | None = None,
    temperature: float | None = None,
    max_repairs: int | None = None,
    timeout: float | None = None,
) -> Callable[..., Any]:
    """Wrap ``fn`` so its body is replaced by an LLM call.

    Supports both ``@llm_function`` and ``@llm_function(tools=[…], model=…)``.
    The wrapped callable preserves ``fn``'s signature and adds support for
    per-call overrides via the same keyword arguments.
    """

    decorator_kwargs = _build_decorator_kwargs(
        model=model,
        cache=cache,
        temperature=temperature,
        max_repairs=max_repairs,
        timeout=timeout,
    )

    def _decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        return _build_wrapper(target, tools=tools, decorator_kwargs=decorator_kwargs)

    if fn is not None:
        return _decorate(fn)
    return _decorate


def _build_decorator_kwargs(
    *,
    model: str | None,
    cache: Literal["on", "off", "replay"] | None,
    temperature: float | None,
    max_repairs: int | None,
    timeout: float | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if model is not None:
        out["model"] = model
    if cache is not None:
        out["cache"] = cache
    if temperature is not None:
        out["temperature"] = temperature
    if max_repairs is not None:
        out["max_repairs"] = max_repairs
    if timeout is not None:
        out["timeout"] = timeout
    return out


def _build_wrapper(
    fn: Callable[..., Any],
    *,
    tools: list[Callable[..., Any]] | Literal["*"] | None,
    decorator_kwargs: dict[str, Any],
) -> Callable[..., Any]:
    """Construct the appropriately-shaped wrapper for ``fn``.

    The wrapper's runtime contract is captured in :class:`_FunctionContext`,
    which is built lazily on first call so import-time errors in the
    docstring don't break the module load.
    """

    sig = inspect.signature(fn)
    is_async = inspect.iscoroutinefunction(fn)
    return_kind = _detect_return_kind(fn)

    ctx_cell: list[_FunctionContext] = []

    def _ensure_ctx() -> _FunctionContext:
        if not ctx_cell:
            ctx_cell.append(
                _FunctionContext(
                    fn=fn,
                    sig=sig,
                    return_kind=return_kind,
                    decorator_kwargs=decorator_kwargs,
                    user_tools_spec=tools,
                )
            )
        return ctx_cell[0]

    if is_async or return_kind.kind != "scalar":
        # Streaming functions are declared with `async def` returning an
        # AsyncIterator, so they always go through the async path.
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = _ensure_ctx()
            return await _run_call(ctx, args, kwargs)

        async_wrapper.__llm_function__ = True  # type: ignore[attr-defined]
        async_wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        async_wrapper.__llm_function_decorator_kwargs__ = decorator_kwargs  # type: ignore[attr-defined]
        async_wrapper.__llm_function_tools_spec__ = tools  # type: ignore[attr-defined]
        async_wrapper.__name__ = fn.__name__
        async_wrapper.__qualname__ = fn.__qualname__
        async_wrapper.__doc__ = fn.__doc__
        async_wrapper.__module__ = fn.__module__
        return async_wrapper

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = _ensure_ctx()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run_call(ctx, args, kwargs))
        raise LLMFunctionError(
            f"Cannot call sync llm_function {fn.__qualname__!r} from inside a running event loop. "
            "Define the function with `async def` and `await` it instead."
        )

    sync_wrapper.__llm_function__ = True  # type: ignore[attr-defined]
    sync_wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    sync_wrapper.__llm_function_decorator_kwargs__ = decorator_kwargs  # type: ignore[attr-defined]
    sync_wrapper.__llm_function_tools_spec__ = tools  # type: ignore[attr-defined]
    sync_wrapper.__name__ = fn.__name__
    sync_wrapper.__qualname__ = fn.__qualname__
    sync_wrapper.__doc__ = fn.__doc__
    sync_wrapper.__module__ = fn.__module__
    return sync_wrapper


# ---------------------------------------------------------------------------
# Return-kind detection.
# ---------------------------------------------------------------------------


class _ReturnKind:
    """Captured return-shape metadata for an :func:`llm_function` callable."""

    __slots__ = ("annotation", "item_type", "kind")

    def __init__(
        self,
        kind: Literal["scalar", "stream_str", "stream_iterable", "stream_partial"],
        annotation: Any,
        item_type: Any = None,
    ) -> None:
        self.kind = kind
        self.annotation = annotation
        self.item_type = item_type


def _detect_return_kind(fn: Callable[..., Any]) -> _ReturnKind:
    annotation = inspect.signature(fn).return_annotation
    try:
        hints = inspect.get_annotations(fn, eval_str=True)
        if "return" in hints:
            annotation = hints["return"]
    except Exception:
        pass

    origin = get_origin(annotation)
    if origin is not None and _is_async_iterator_origin(origin):
        type_args = get_args(annotation)
        item = type_args[0] if type_args else str
        if item is str:
            return _ReturnKind("stream_str", annotation, str)
        if isinstance(item, type) and issubclass(item, BaseModel):
            return _ReturnKind("stream_partial", annotation, item)
        return _ReturnKind("stream_iterable", annotation, item)

    return _ReturnKind("scalar", annotation)


def _is_async_iterator_origin(origin: Any) -> bool:
    from collections.abc import AsyncIterator as _AbcAsyncIterator

    return origin is AsyncIterator or origin is _AbcAsyncIterator


# ---------------------------------------------------------------------------
# Per-function context (built once, reused across calls).
# ---------------------------------------------------------------------------


class _FunctionContext:
    """All static-per-function data the runtime pipeline needs."""

    def __init__(
        self,
        *,
        fn: Callable[..., Any],
        sig: inspect.Signature,
        return_kind: _ReturnKind,
        decorator_kwargs: dict[str, Any],
        user_tools_spec: list[Callable[..., Any]] | Literal["*"] | None,
    ) -> None:
        self.fn = fn
        self.sig = sig
        self.return_kind = return_kind
        self.decorator_kwargs = decorator_kwargs
        self.user_tools_spec = user_tools_spec

        self.docstring: DocstringSpec = parse_docstring(fn)
        self.input_schema: dict[str, Any] = build_input_schema(fn, docstring=self.docstring)
        self.output_was_enveloped: bool = False
        if return_kind.kind == "scalar":
            raw_output = build_output_schema(fn, docstring=self.docstring)
            envelope, was_wrapped = _ensure_object_schema(raw_output)
            self.output_schema: dict[str, Any] = envelope
            self.output_was_enveloped = was_wrapped
        else:
            self.output_schema = {}

        # Output adapter: validate provider dicts into the user's return type.
        # For streaming we don't construct one here — _streaming handles it.
        self.output_adapter: TypeAdapter[Any] | None
        if return_kind.kind == "scalar":
            return_annotation = return_kind.annotation
            if return_annotation is inspect.Signature.empty:
                return_annotation = Any
            try:
                self.output_adapter = TypeAdapter(return_annotation)
            except Exception:
                self.output_adapter = None
        else:
            self.output_adapter = None

        # Map: which signature param keys are real function args.
        self.param_names: set[str] = {
            name
            for name, param in sig.parameters.items()
            if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }


# ---------------------------------------------------------------------------
# Per-call pipeline.
# ---------------------------------------------------------------------------


async def _run_call(
    ctx: _FunctionContext,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    debug = bool(kwargs.pop("debug", False))
    call_kwargs, fn_kwargs = _split_per_call_kwargs(kwargs)
    bound_args = _bind_args(ctx, args, fn_kwargs)
    arg_dict = dict(bound_args.arguments)

    settings = resolve(
        decorator=ctx.decorator_kwargs,
        docstring=_docstring_to_settings_layer(ctx.docstring),
        call=call_kwargs,
    )

    user_tools = resolve_tools(ctx.user_tools_spec, settings)
    raise_tool_schemas = synthesise_raise_tools(ctx.docstring.raises)

    system_prompt = _build_system_prompt(ctx)
    user_message = _build_user_message(ctx, arg_dict)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    sampling = {
        "temperature": settings.temperature,
        "max_repairs": settings.max_repairs,
        "max_tool_iterations": settings.max_tool_iterations,
    }
    tool_schemas = [t.to_openai() for t in user_tools] + raise_tool_schemas
    cache_key = make_key(
        fn=ctx.fn,
        model=settings.model,
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        args=arg_dict,
        kwargs={},
        sampling=sampling,
    )

    if debug:
        _print_debug_dump(
            ctx=ctx,
            settings=settings,
            messages=messages,
            tool_schemas=tool_schemas,
            output_schema=ctx.output_schema,
        )

    if ctx.return_kind.kind != "scalar":
        return await _run_streaming(
            ctx,
            settings=settings,
            messages=messages,
            cache_key=cache_key,
        )

    cached = _try_cached(settings, cache_key)
    if cached is not MISS:
        return _coerce_output(ctx, cached)

    result_dict = await _run_provider(
        ctx,
        settings=settings,
        messages=messages,
        user_tools=user_tools,
        raise_tool_schemas=raise_tool_schemas,
    )

    if settings.cache == "on":
        with ResultCache(settings) as cache:
            cache.set(cache_key, result_dict)

    return _coerce_output(ctx, result_dict)


def _try_cached(settings: Settings, cache_key: Any) -> Any:
    if settings.cache == "on":
        with ResultCache(settings) as cache:
            value = cache.get(cache_key)
            if value is not MISS:
                return value
    if settings.cache == "replay":
        store = ReplayStore(settings)
        return store.get(cache_key)
    return MISS


async def _run_provider(
    ctx: _FunctionContext,
    *,
    settings: Settings,
    messages: list[dict[str, Any]],
    user_tools: list[ToolDef],
    raise_tool_schemas: list[dict[str, Any]],
) -> dict[str, Any]:
    provider = _provider_factory()
    raise_lookup = {f"raise_{spec.name}": spec.exc_type for spec in ctx.docstring.raises}
    invoker = _make_tool_invoker(user_tools, raise_lookup)

    tool_payload: list[dict[str, Any]] = [t.to_openai() for t in user_tools]
    tool_payload.extend(raise_tool_schemas)

    api_key, api_base, extra_headers = _select_provider(settings)

    try:
        return await provider.complete(
            messages=messages,
            output_schema=ctx.output_schema,
            model=settings.model,
            temperature=settings.temperature,
            timeout=settings.timeout,
            max_repairs=settings.max_repairs,
            max_tool_iterations=settings.max_tool_iterations,
            tools=tool_payload or None,
            tool_invoker=invoker if tool_payload else None,
            api_key=api_key,
            api_base=api_base,
            extra_headers=extra_headers,
        )
    except _LLMFunctionRaisedException as raised:
        raise raised.exc_type(raised.reason) from None


def _select_provider(
    settings: Settings,
) -> tuple[str | None, str | None, dict[str, str] | None]:
    """Pick the configured ``ProviderConfig`` for ``settings.model``.

    Matches the ``provider/`` prefix in ``settings.providers``. Returns
    ``(api_key, api_base, extra_headers)`` — any element may be ``None`` when
    that field is unset; an entirely-unconfigured provider returns
    ``(None, None, None)`` and the upstream call falls back to environment
    variables.
    """

    providers = settings.providers
    if not providers:
        return None, None, None
    model = settings.model or ""
    prefix = model.split("/", 1)[0] if "/" in model else model
    entry = providers.get(prefix)
    if entry is None:
        lower = prefix.lower()
        for k, v in providers.items():
            if k.lower() == lower:
                entry = v
                break
    if entry is None:
        return None, None, None
    return (
        entry.resolve_api_key(),
        entry.api_base,
        dict(entry.extra_headers) if entry.extra_headers else None,
    )


def _make_tool_invoker(
    user_tools: list[ToolDef],
    raise_lookup: dict[str, type[BaseException]],
) -> Callable[[str, dict[str, Any]], Awaitable[Any]]:
    async def _invoker(name: str, args: dict[str, Any]) -> Any:
        if name in raise_lookup:
            reason = str(args.get("reason", "")).strip() or "(no reason given)"
            raise _LLMFunctionRaisedException(raise_lookup[name], reason)
        return await invoke_tool(name, args, user_tools)

    return _invoker


# ---------------------------------------------------------------------------
# Streaming pipeline.
# ---------------------------------------------------------------------------


async def _run_streaming(
    ctx: _FunctionContext,
    *,
    settings: Settings,
    messages: list[dict[str, Any]],
    cache_key: Any,
) -> Any:
    """Drive a streaming completion to its final assembled value.

    On a cache hit we re-emit the cached value as a single chunk. On a miss
    we materialise the entire stream into a final value, cache it, and then
    return an iterator that yields that value once. This is documented
    behaviour — chunk-level caching is intentionally unsupported.
    """

    if settings.cache == "on":
        with ResultCache(settings) as cache:
            cached = cache.get(cache_key)
            if cached is not MISS:
                return _replay_stream(ctx, cached)

    if settings.cache == "replay":
        store = ReplayStore(settings)
        cached = store.get(cache_key)
        return _replay_stream(ctx, cached)

    provider = _provider_factory()
    api_key, api_base, extra_headers = _select_provider(settings)
    delta_stream = provider.complete_stream(
        messages=messages,
        model=settings.model,
        temperature=settings.temperature,
        timeout=settings.timeout,
        return_kind=cast(
            Literal["str", "iterable", "partial"],
            {
                "stream_str": "str",
                "stream_iterable": "iterable",
                "stream_partial": "partial",
            }[ctx.return_kind.kind],
        ),
        output_schema=ctx.output_schema or None,
        api_key=api_key,
        api_base=api_base,
        extra_headers=extra_headers,
    )

    materialised, async_iter = _wrap_stream(ctx, delta_stream)

    async def _emit() -> AsyncIterator[Any]:
        async for chunk in async_iter:
            yield chunk
        if settings.cache == "on":
            with ResultCache(settings) as cache:
                cache.set(cache_key, materialised.value)

    return _emit()


def _wrap_stream(
    ctx: _FunctionContext,
    delta_stream: AsyncIterator[str],
) -> tuple[_StreamMaterialiser, AsyncIterator[Any]]:
    """Tee the provider stream through the appropriate shaper.

    The materialiser captures every emission so the value can be cached after
    the iterator drains.
    """

    materialised = _StreamMaterialiser(kind=ctx.return_kind.kind)

    if ctx.return_kind.kind == "stream_str":

        async def gen_str() -> AsyncIterator[str]:
            async for piece in stream_str(delta_stream):
                materialised.add(piece)
                yield piece

        return materialised, gen_str()

    if ctx.return_kind.kind == "stream_iterable":
        item_type: Any = ctx.return_kind.item_type or Any

        async def gen_items() -> AsyncIterator[Any]:
            iterator: AsyncIterator[Any] = stream_iterable(delta_stream, cast(type, item_type))
            async for item in iterator:
                materialised.add(item)
                yield item

        return materialised, gen_items()

    # stream_partial
    model_type = ctx.return_kind.item_type
    if not (isinstance(model_type, type) and issubclass(model_type, BaseModel)):
        raise ConfigurationError(
            "AsyncIterator partial streaming requires a Pydantic BaseModel item type."
        )

    async def gen_partials() -> AsyncIterator[BaseModel]:
        async for partial in stream_partial(delta_stream, model_type):
            materialised.add(partial)
            yield partial

    return materialised, gen_partials()


class _StreamMaterialiser:
    """Captures the items produced by a streaming wrapper for caching."""

    def __init__(self, *, kind: str) -> None:
        self._kind = kind
        self._buffer_str: list[str] = []
        self._buffer_items: list[Any] = []

    def add(self, item: Any) -> None:
        if self._kind == "stream_str":
            self._buffer_str.append(cast(str, item))
        else:
            self._buffer_items.append(item)

    @property
    def value(self) -> Any:
        if self._kind == "stream_str":
            return "".join(self._buffer_str)
        if self._kind == "stream_iterable":
            return [_jsonable(item) for item in self._buffer_items]
        # stream_partial: cache the final model as a dict.
        if self._buffer_items:
            tail = self._buffer_items[-1]
            if isinstance(tail, BaseModel):
                return tail.model_dump()
            return tail
        return None


def _replay_stream(ctx: _FunctionContext, cached: Any) -> AsyncIterator[Any]:
    """Re-emit a cached streaming value as a single-chunk iterator."""

    kind = ctx.return_kind.kind

    async def emit() -> AsyncIterator[Any]:
        if kind == "stream_str":
            yield cast(str, cached)
            return
        if kind == "stream_iterable":
            item_type = ctx.return_kind.item_type
            if isinstance(cached, list):
                if item_type is not None and item_type is not Any:
                    adapter: TypeAdapter[Any] = TypeAdapter(item_type)
                    for raw in cached:
                        yield adapter.validate_python(raw)
                else:
                    for raw in cached:
                        yield raw
            else:
                yield cached
            return
        # stream_partial
        model_type = ctx.return_kind.item_type
        if isinstance(model_type, type) and issubclass(model_type, BaseModel):
            yield model_type.model_validate(cached)
        else:
            yield cached

    return emit()


def _jsonable(item: Any) -> Any:
    if isinstance(item, BaseModel):
        return item.model_dump()
    return item


def _print_debug_dump(
    *,
    ctx: _FunctionContext,
    settings: Settings,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    output_schema: dict[str, Any],
) -> None:
    """Pretty-print the rendered prompt for a single call to stderr.

    Triggered by passing ``debug=True`` at call time. Independent of the
    library's logging configuration so it works without setup. Output goes to
    ``sys.stderr`` to keep ``stdout`` clean for callers piping results.
    """

    import sys

    out = sys.stderr
    name = ctx.fn.__qualname__
    bar = "=" * 72
    out.write(f"\n{bar}\nllm_functions debug: {name}\n{bar}\n")
    out.write(f"model:       {settings.model}\n")
    out.write(f"temperature: {settings.temperature}\n")
    out.write(f"cache:       {settings.cache}\n")
    if settings.providers:
        provider_prefix = (
            settings.model.split("/", 1)[0] if "/" in settings.model else settings.model
        )
        if provider_prefix in settings.providers:
            entry = settings.providers[provider_prefix]
            if entry.api_base:
                out.write(f"api_base:    {entry.api_base}\n")
    for msg in messages:
        role = msg.get("role", "?")
        out.write(f"\n--- {role} ---\n")
        out.write(str(msg.get("content", "")))
        out.write("\n")
    if tool_schemas:
        names = [t.get("function", {}).get("name", "?") for t in tool_schemas]
        out.write(f"\n--- tools ---\n{', '.join(names)}\n")
    if output_schema:
        out.write("\n--- output schema ---\n")
        out.write(json.dumps(output_schema, indent=2, sort_keys=True))
        out.write("\n")
    out.write(f"{bar}\n")
    out.flush()


# ---------------------------------------------------------------------------
# Argument handling, prompt building, output coercion.
# ---------------------------------------------------------------------------


def _split_per_call_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split ``kwargs`` into ``(per_call, function_args)``.

    Per-call kwargs (``cache``, ``model``, ``temperature``, ``max_repairs``,
    ``timeout``) are consumed by the decorator pipeline. Everything else is
    forwarded to the LLM as the user's actual arguments.
    """

    per_call: dict[str, Any] = {}
    fn_args: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in _PER_CALL_KWARGS:
            per_call[key] = value
        else:
            fn_args[key] = value
    return per_call, fn_args


def _bind_args(
    ctx: _FunctionContext,
    args: tuple[Any, ...],
    fn_kwargs: dict[str, Any],
) -> inspect.BoundArguments:
    try:
        bound = ctx.sig.bind(*args, **fn_kwargs)
    except TypeError as exc:
        raise ConfigurationError(
            f"Arguments to {ctx.fn.__qualname__!r} did not match its signature: {exc}"
        ) from exc
    bound.apply_defaults()
    return bound


def _docstring_to_settings_layer(docstring: DocstringSpec) -> dict[str, Any]:
    layer: dict[str, Any] = {}
    if docstring.model is not None:
        layer["model"] = docstring.model
    if docstring.cache is not None:
        layer["cache"] = docstring.cache
    return layer


def _build_system_prompt(ctx: _FunctionContext) -> str:
    """Compose a system prompt from the parsed docstring sections."""

    spec = ctx.docstring
    lines: list[str] = []
    name = ctx.fn.__qualname__
    lines.append(
        f"You are the implementation of the Python function `{name}`. "
        "Follow its docstring contract precisely."
    )
    if spec.task:
        lines.append("\nTask:")
        lines.append(spec.task.strip())
    if spec.context:
        lines.append("\nContext:")
        lines.append(spec.context.strip())
    if spec.constraints:
        lines.append("\nConstraints:")
        for item in spec.constraints:
            lines.append(f"- {item}")
    if spec.format_hints:
        lines.append("\nFormat hints:")
        lines.append(spec.format_hints.strip())
    if spec.tools_guidance:
        lines.append("\nTools guidance:")
        for tool_name, guidance in spec.tools_guidance.items():
            lines.append(f"- {tool_name}: {guidance}")
    if spec.raises:
        lines.append("\nFailure modes:")
        for raise_spec in spec.raises:
            lines.append(
                f"- If {raise_spec.when or 'the precondition fails'}, "
                f"call the tool `raise_{raise_spec.name}` with a short reason "
                f"instead of returning a value."
            )
    if spec.notes:
        lines.append("\nNotes:")
        lines.append(spec.notes.strip())
    return "\n".join(lines).strip()


def _build_user_message(ctx: _FunctionContext, arg_values: dict[str, Any]) -> str:
    """Render the call site's argument values as a user message."""

    if not arg_values:
        return f"Call {ctx.fn.__name__}() with no arguments."
    payload = json.dumps(_jsonable_args(arg_values), indent=2, sort_keys=True, default=str)
    return f"Call {ctx.fn.__name__} with these arguments:\n{payload}"


def _jsonable_args(arg_values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in arg_values.items():
        if isinstance(value, BaseModel):
            out[key] = value.model_dump()
        else:
            out[key] = value
    return out


def _coerce_output(ctx: _FunctionContext, raw: Any) -> Any:
    """Validate ``raw`` against the function's return annotation.

    Provider returns ``dict``-shaped payloads even for scalar return types
    (the dict has a ``{"value": …}`` envelope when the user's return type is
    not itself an object). We unwrap that here before validation.
    """

    annotation = ctx.sig.return_annotation
    if annotation is inspect.Signature.empty or annotation is None:
        return raw

    adapter = ctx.output_adapter
    if adapter is None:
        return raw

    if ctx.output_was_enveloped and isinstance(raw, dict) and "value" in raw:
        payload: Any = raw["value"]
    else:
        payload = _unwrap_value_envelope(raw, annotation)
    try:
        return adapter.validate_python(payload)
    except Exception:
        if isinstance(raw, dict) and "value" in raw and payload is not raw["value"]:
            return adapter.validate_python(raw["value"])
        raise


def _unwrap_value_envelope(raw: Any, annotation: Any) -> Any:
    """Strip a ``{"value": …}`` wrapper if the return type isn't a dict/model.

    The provider asks the model for a JSON object matching ``output_schema``;
    when the user's return type is scalar (``int``, ``str``, ``list``, …) we
    wrap it in ``{"value": <scalar>}`` server-side via the schema and unwrap
    it here. For Pydantic-model returns we pass the dict through unchanged.
    """

    if isinstance(raw, dict) and set(raw.keys()) == {"value"}:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return raw
        return raw["value"]
    return raw


def _ensure_object_schema(schema: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Wrap a non-object output schema in a ``{"value": <schema>}`` envelope.

    OpenAI/LiteLLM tool-call ``parameters`` must be ``type: object``. When the
    user declares a scalar/list return type, we wrap the schema so the model
    emits ``{"value": <thing>}`` and the decorator unwraps that on the way
    out. Returns ``(envelope_schema, was_wrapped)``.
    """

    schema_type = schema.get("type")
    if schema_type == "object" and "properties" in schema:
        return schema, False

    inner = {k: v for k, v in schema.items() if k not in {"title"}}
    title = schema.get("title", "output")
    description = schema.pop("description", None) if isinstance(schema, dict) else None
    inner_no_desc = {k: v for k, v in inner.items() if k != "description"}
    envelope: dict[str, Any] = {
        "type": "object",
        "title": title,
        "properties": {"value": inner_no_desc},
        "required": ["value"],
        "additionalProperties": False,
    }
    if description:
        envelope["description"] = description
    return envelope, True
