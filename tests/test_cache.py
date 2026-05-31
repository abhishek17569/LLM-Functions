"""Tests for llmfunctionkit._cache."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from llmfunctionkit._cache import (
    MISS,
    CacheKey,
    ResultCache,
    make_key,
    resolve_cache_dir,
)
from llmfunctionkit._config import configure, reset_settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _baseline_fn(x: int, y: int = 0) -> int:
    """A reference function whose source is hashed into the cache key."""

    return x + y


def _other_fn(x: int, y: int = 0) -> int:
    """A second function with intentionally different source."""

    return x * y + 1


@pytest.fixture(autouse=True)
def _reset_settings() -> Iterator[None]:
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    target = tmp_path / "llmfunctionkit_cache"
    configure(cache_dir=str(target))
    return target


@pytest.fixture
def baseline_kwargs() -> dict[str, Any]:
    return {
        "fn": _baseline_fn,
        "model": "openai/gpt-4o-mini",
        "system_prompt": "you are helpful",
        "tool_schemas": [{"name": "lookup", "parameters": {"type": "object"}}],
        "args": {"x": 1},
        "kwargs": {"y": 2},
        "sampling": {"temperature": 0.0, "top_p": 1.0},
    }


def _replace(base: dict[str, Any], **changes: Any) -> dict[str, Any]:
    out = dict(base)
    out.update(changes)
    return out


# ---------------------------------------------------------------------------
# Key stability — one test per varying component, plus the all-stable baseline.
# ---------------------------------------------------------------------------


def test_key_stable_when_nothing_changes(baseline_kwargs: dict[str, Any]) -> None:
    a = make_key(**baseline_kwargs).to_sha256()
    b = make_key(**baseline_kwargs).to_sha256()
    assert a == b


def test_key_changes_when_function_source_changes(baseline_kwargs: dict[str, Any]) -> None:
    base_digest = make_key(**baseline_kwargs).to_sha256()
    other_digest = make_key(**_replace(baseline_kwargs, fn=_other_fn)).to_sha256()
    assert base_digest != other_digest


def test_key_changes_when_model_changes(baseline_kwargs: dict[str, Any]) -> None:
    base = make_key(**baseline_kwargs).to_sha256()
    other = make_key(**_replace(baseline_kwargs, model="openai/gpt-4o")).to_sha256()
    assert base != other


def test_key_changes_when_system_prompt_changes(baseline_kwargs: dict[str, Any]) -> None:
    base = make_key(**baseline_kwargs).to_sha256()
    other = make_key(**_replace(baseline_kwargs, system_prompt="different")).to_sha256()
    assert base != other


def test_key_changes_when_tool_schemas_change(baseline_kwargs: dict[str, Any]) -> None:
    base = make_key(**baseline_kwargs).to_sha256()
    other = make_key(
        **_replace(baseline_kwargs, tool_schemas=[{"name": "other", "parameters": {}}])
    ).to_sha256()
    assert base != other


def test_key_changes_when_args_change(baseline_kwargs: dict[str, Any]) -> None:
    base = make_key(**baseline_kwargs).to_sha256()
    other = make_key(**_replace(baseline_kwargs, args={"x": 99})).to_sha256()
    assert base != other


def test_key_changes_when_kwargs_change(baseline_kwargs: dict[str, Any]) -> None:
    base = make_key(**baseline_kwargs).to_sha256()
    other = make_key(**_replace(baseline_kwargs, kwargs={"y": 99})).to_sha256()
    assert base != other


def test_key_changes_when_sampling_changes(baseline_kwargs: dict[str, Any]) -> None:
    base = make_key(**baseline_kwargs).to_sha256()
    other = make_key(**_replace(baseline_kwargs, sampling={"temperature": 0.7})).to_sha256()
    assert base != other


def test_key_changes_when_library_version_changes(
    monkeypatch: pytest.MonkeyPatch, baseline_kwargs: dict[str, Any]
) -> None:
    base = make_key(**baseline_kwargs)
    monkeypatch.setattr("llmfunctionkit._cache._library_version", lambda: "9.9.9")
    bumped = make_key(**baseline_kwargs)
    assert base.to_sha256() != bumped.to_sha256()
    assert bumped.library_version == "9.9.9"


# ---------------------------------------------------------------------------
# Key normalisation invariants.
# ---------------------------------------------------------------------------


def test_key_is_independent_of_dict_ordering(baseline_kwargs: dict[str, Any]) -> None:
    a = make_key(**_replace(baseline_kwargs, sampling={"temperature": 0.0, "top_p": 1.0}))
    b = make_key(**_replace(baseline_kwargs, sampling={"top_p": 1.0, "temperature": 0.0}))
    assert a.to_sha256() == b.to_sha256()


def test_key_handles_missing_source_gracefully() -> None:
    # builtins have no inspect.getsource — should not crash.
    key = make_key(
        fn=len,
        model="m",
        system_prompt="",
        tool_schemas=[],
        args={"x": [1, 2, 3]},
        kwargs={},
        sampling={},
    )
    assert isinstance(key.to_sha256(), str)
    assert len(key.to_sha256()) == 64


def test_library_version_falls_back_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib.metadata import PackageNotFoundError

    def _raise(_: str) -> str:
        raise PackageNotFoundError("llmfunctionkit")

    monkeypatch.setattr("llmfunctionkit._cache.version", _raise)
    from llmfunctionkit._cache import _library_version  # type: ignore[attr-defined]

    assert _library_version() == "0.0.0+dev"


# ---------------------------------------------------------------------------
# ResultCache: hit / miss / disable / enable.
# ---------------------------------------------------------------------------


def test_cache_get_returns_miss_when_empty(
    cache_dir: Path, baseline_kwargs: dict[str, Any]
) -> None:
    key = make_key(**baseline_kwargs)
    with ResultCache() as cache:
        assert cache.get(key) is MISS


def test_cache_set_then_get_returns_value(cache_dir: Path, baseline_kwargs: dict[str, Any]) -> None:
    key = make_key(**baseline_kwargs)
    with ResultCache() as cache:
        cache.set(key, {"answer": 42})
        assert cache.get(key) == {"answer": 42}


def test_cache_disable_blocks_reads_and_writes(
    cache_dir: Path, baseline_kwargs: dict[str, Any]
) -> None:
    key = make_key(**baseline_kwargs)
    with ResultCache() as cache:
        cache.set(key, "stored")
        cache.disable()
        assert cache.get(key) is MISS
        cache.set(key, "ignored")
        cache.enable()
        # Original value was never overwritten because cache was disabled on set.
        assert cache.get(key) == "stored"


def test_cache_enable_after_disable_resumes_reads(
    cache_dir: Path, baseline_kwargs: dict[str, Any]
) -> None:
    key = make_key(**baseline_kwargs)
    with ResultCache() as cache:
        cache.set(key, "stored")
        cache.disable()
        cache.enable()
        assert cache.get(key) == "stored"
        assert cache.enabled is True


def test_cache_clear_removes_all_entries(cache_dir: Path, baseline_kwargs: dict[str, Any]) -> None:
    key = make_key(**baseline_kwargs)
    with ResultCache() as cache:
        cache.set(key, "stored")
        cache.clear()
        assert cache.get(key) is MISS


def test_miss_is_falsy_singleton() -> None:
    from llmfunctionkit._cache import _Miss  # type: ignore[attr-defined]

    assert bool(MISS) is False
    assert _Miss() is MISS
    assert repr(MISS) == "<llm_function.MISS>"


# ---------------------------------------------------------------------------
# Cache directory resolution.
# ---------------------------------------------------------------------------


def test_resolve_cache_dir_uses_settings_cache_dir(tmp_path: Path) -> None:
    target = tmp_path / "explicit"
    configure(cache_dir=str(target))
    resolved = resolve_cache_dir()
    assert resolved == target
    assert resolved.is_dir()


def test_resolve_cache_dir_uses_xdg_cache_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    reset_settings()
    resolved = resolve_cache_dir()
    assert resolved == tmp_path / "llmfunctionkit"
    assert resolved.is_dir()


def test_resolve_cache_dir_falls_back_to_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))  # type: ignore[arg-type]
    reset_settings()
    resolved = resolve_cache_dir()
    assert resolved == tmp_path / ".cache" / "llmfunctionkit"
    assert resolved.is_dir()


def test_cache_key_is_frozen() -> None:
    from pydantic import ValidationError

    key = CacheKey(
        function_source_hash="a" * 64,
        model="m",
        system_prompt="",
        tool_schemas_hash="b" * 64,
        args_hash="c" * 64,
        sampling_params_hash="d" * 64,
        library_version="0.1.0",
    )
    with pytest.raises(ValidationError):
        key.model = "other"  # type: ignore[misc]
