"""Content-addressed result cache for llm_function.

The cache key is the SHA256 of a Pydantic :class:`CacheKey` containing seven
components: function source, model, system prompt, tool schema definitions,
normalised call arguments, sampling parameters, and library version. Changing
any one of these invalidates the entry; changing none keeps the same key.

Storage backend is :mod:`diskcache`, rooted at ``Settings.cache_dir`` if set,
otherwise ``$XDG_CACHE_HOME/llm_function`` (falling back to
``~/.cache/llm_function``).

Streaming note: callers that stream tokens MUST cache the *final assembled
value* (the deserialised return type), not the chunks. Chunk-level caching is
intentionally unsupported because chunk shape is provider-specific and not
part of the public contract.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final

import diskcache  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict
from pydantic_core import to_jsonable_python

from ._config import Settings, get_settings

__all__ = [
    "MISS",
    "CacheKey",
    "ResultCache",
    "make_key",
    "resolve_cache_dir",
]


class _Miss:
    """Sentinel returned by :meth:`ResultCache.get` for cache misses."""

    _instance: _Miss | None = None

    def __new__(cls) -> _Miss:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<llm_function.MISS>"

    def __bool__(self) -> bool:
        return False


MISS: Final[_Miss] = _Miss()


def _library_version() -> str:
    try:
        return version("llmfunctionkit")
    except PackageNotFoundError:
        return "0.0.0+dev"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_jsonable(obj: Any) -> str:
    """Hash any JSON-able object via Pydantic's normalisation + sorted JSON."""

    normalised = to_jsonable_python(obj)
    serialised = json.dumps(normalised, sort_keys=True, separators=(",", ":"))
    return _hash_text(serialised)


class CacheKey(BaseModel):
    """Seven content-addressed components that identify a cacheable call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    function_source_hash: str
    model: str
    system_prompt: str
    tool_schemas_hash: str
    args_hash: str
    sampling_params_hash: str
    library_version: str

    def to_sha256(self) -> str:
        """Return a stable SHA256 hex digest of this key."""

        payload = self.model_dump()
        serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return _hash_text(serialised)


def make_key(
    *,
    fn: Callable[..., Any],
    model: str,
    system_prompt: str,
    tool_schemas: list[dict[str, Any]],
    args: dict[str, Any],
    kwargs: dict[str, Any],
    sampling: dict[str, Any],
) -> CacheKey:
    """Build a :class:`CacheKey` for one call.

    The function source hash uses :func:`inspect.getsource`. Args and kwargs
    are merged into a single mapping ``{"args": args, "kwargs": kwargs}``,
    normalised through :func:`pydantic_core.to_jsonable_python`, and hashed
    via canonical sorted-key JSON. Tool schemas and sampling params follow
    the same normalisation.
    """

    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        # Builtins / dynamically-generated functions have no source. Fall
        # back to a stable identifier that still varies per function.
        source = f"<no-source:{getattr(fn, '__qualname__', repr(fn))}>"

    return CacheKey(
        function_source_hash=_hash_text(source),
        model=model,
        system_prompt=system_prompt,
        tool_schemas_hash=_hash_jsonable(tool_schemas),
        args_hash=_hash_jsonable({"args": args, "kwargs": kwargs}),
        sampling_params_hash=_hash_jsonable(sampling),
        library_version=_library_version(),
    )


def resolve_cache_dir(settings: Settings | None = None) -> Path:
    """Return the on-disk cache directory.

    Precedence: ``settings.cache_dir`` > ``$XDG_CACHE_HOME/llmfunctionkit`` >
    ``~/.cache/llmfunctionkit``. The directory is created if it does not exist.
    """

    if settings is None:
        settings = get_settings()
    if settings.cache_dir:
        path = Path(settings.cache_dir).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
        path = base / "llmfunctionkit"
    path.mkdir(parents=True, exist_ok=True)
    return path


class ResultCache:
    """Thin wrapper around :class:`diskcache.Cache`.

    The cache is keyed by :meth:`CacheKey.to_sha256` to keep on-disk paths
    short and predictable. Use :meth:`disable` / :meth:`enable` to bypass
    reads and writes without tearing down the underlying store.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._directory = resolve_cache_dir(settings)
        self._cache: diskcache.Cache = diskcache.Cache(str(self._directory))
        self._enabled: bool = True

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def enabled(self) -> bool:
        return self._enabled

    def disable(self) -> None:
        """Stop reading from / writing to the cache. Existing entries persist."""

        self._enabled = False

    def enable(self) -> None:
        """Resume reading from / writing to the cache."""

        self._enabled = True

    def get(self, key: CacheKey) -> Any:
        """Return the cached value or :data:`MISS` if absent / disabled."""

        if not self._enabled:
            return MISS
        digest = key.to_sha256()
        sentinel = object()
        value = self._cache.get(digest, default=sentinel)
        if value is sentinel:
            return MISS
        return value

    def set(self, key: CacheKey, value: Any) -> None:
        """Store ``value`` under ``key``. No-op when disabled."""

        if not self._enabled:
            return
        self._cache.set(key.to_sha256(), value)

    def clear(self) -> None:
        """Remove all entries. Used by tests."""

        self._cache.clear()

    def close(self) -> None:
        """Release the underlying diskcache handle."""

        self._cache.close()

    def __enter__(self) -> ResultCache:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
