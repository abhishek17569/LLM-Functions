"""Tests for llmfunctionkit._replay."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from llmfunctionkit._cache import CacheKey, make_key
from llmfunctionkit._config import configure, reset_settings
from llmfunctionkit._replay import (
    RECORD_ENV_VAR,
    ReplayMissError,
    ReplayStore,
    is_recording,
)


def _baseline_fn(x: int) -> int:
    return x


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(RECORD_ENV_VAR, raising=False)
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    target = tmp_path / "fixtures"
    configure(replay_fixtures_dir=str(target))
    return target


@pytest.fixture
def cache_key() -> CacheKey:
    return make_key(
        fn=_baseline_fn,
        model="m",
        system_prompt="",
        tool_schemas=[],
        args={"x": 1},
        kwargs={},
        sampling={},
    )


def test_replay_miss_raises_with_suggested_path(fixtures_dir: Path, cache_key: CacheKey) -> None:
    store = ReplayStore()
    with pytest.raises(ReplayMissError) as exc_info:
        store.get(cache_key)
    assert exc_info.value.key.to_sha256() == cache_key.to_sha256()
    assert exc_info.value.suggested_path == fixtures_dir / f"{cache_key.to_sha256()}.json"
    assert RECORD_ENV_VAR in str(exc_info.value)
    assert str(fixtures_dir) in str(exc_info.value)


def test_replay_record_writes_json_when_recording(
    fixtures_dir: Path,
    cache_key: CacheKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RECORD_ENV_VAR, "1")
    store = ReplayStore()
    payload: dict[str, Any] = {"answer": 42, "items": [1, 2, 3]}
    path = store.record(cache_key, payload)
    assert path.is_file()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["value"] == payload
    assert on_disk["key"]["model"] == cache_key.model


def test_replay_record_is_noop_without_env_var(fixtures_dir: Path, cache_key: CacheKey) -> None:
    store = ReplayStore()
    path = store.record(cache_key, "ignored")
    assert not path.is_file()
    assert path == fixtures_dir / f"{cache_key.to_sha256()}.json"


def test_replay_record_then_get_round_trips(
    fixtures_dir: Path,
    cache_key: CacheKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RECORD_ENV_VAR, "1")
    store = ReplayStore()
    store.record(cache_key, {"hello": "world"})
    assert store.get(cache_key) == {"hello": "world"}
    assert store.has(cache_key) is True


def test_replay_get_raises_when_file_is_corrupt(fixtures_dir: Path, cache_key: CacheKey) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    path = fixtures_dir / f"{cache_key.to_sha256()}.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    store = ReplayStore()
    with pytest.raises(ReplayMissError):
        store.get(cache_key)


def test_replay_get_raises_when_payload_missing_value_key(
    fixtures_dir: Path, cache_key: CacheKey
) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    path = fixtures_dir / f"{cache_key.to_sha256()}.json"
    path.write_text(json.dumps({"unrelated": 1}), encoding="utf-8")
    store = ReplayStore()
    with pytest.raises(ReplayMissError):
        store.get(cache_key)


def test_replay_path_for_uses_sha256_digest(fixtures_dir: Path, cache_key: CacheKey) -> None:
    store = ReplayStore()
    path = store.path_for(cache_key)
    assert path.name.endswith(".json")
    assert cache_key.to_sha256() in path.name


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("", False),
        ("false", False),
        ("nope", False),
    ],
)
def test_is_recording_handles_truthy_strings(
    raw: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if raw == "":
        monkeypatch.delenv(RECORD_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(RECORD_ENV_VAR, raw)
    assert is_recording() is expected


def test_replay_store_directory_property(fixtures_dir: Path) -> None:
    store = ReplayStore()
    assert store.directory == fixtures_dir


def test_replay_record_uses_atomic_replace(
    fixtures_dir: Path,
    cache_key: CacheKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording must not leave .tmp turds behind on success."""

    monkeypatch.setenv(RECORD_ENV_VAR, "1")
    store = ReplayStore()
    store.record(cache_key, {"v": 1})
    leftover = list(fixtures_dir.glob("*.tmp"))
    assert leftover == []
