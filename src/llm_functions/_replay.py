"""Replay-from-fixture store for llm_function.

A separate execution path from :mod:`llm_functions._cache`. Where the result
cache is opportunistic and content-addressed on disk, the replay store is
deterministic and tracked in the source tree.

Modes
-----
* ``cache="replay"`` (read-only): :meth:`ReplayStore.get` returns the fixture
  if present, else raises :class:`ReplayMissError` with the suggested fixture
  path so the user knows where to drop the recorded JSON.
* ``cache="on"`` with ``LLM_FUNCTIONS_RECORD=1`` (write mode): callers may use
  :meth:`ReplayStore.record` to write a fresh fixture for the given key.
  Without the env var, ``record`` is a no-op so test runs do not silently
  mutate fixtures.

Fixtures are JSON files, one per ``CacheKey.to_sha256()`` digest, stored at
``Settings.replay_fixtures_dir``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ._cache import CacheKey
from ._config import LLMFunctionError, Settings, get_settings

__all__ = [
    "RECORD_ENV_VAR",
    "ReplayMissError",
    "ReplayStore",
    "is_recording",
]


RECORD_ENV_VAR: str = "LLM_FUNCTIONS_RECORD"


def is_recording() -> bool:
    """Return ``True`` when ``LLM_FUNCTIONS_RECORD`` is set to a truthy value."""

    raw = os.environ.get(RECORD_ENV_VAR, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class ReplayMissError(LLMFunctionError):
    """Raised when ``cache="replay"`` cannot find a fixture for a key.

    Attributes:
        key: The :class:`CacheKey` whose fixture is missing.
        suggested_path: Where the fixture should be written. Surfaced in the
            error message so users can rerun with ``LLM_FUNCTIONS_RECORD=1`` to
            populate it.
    """

    def __init__(self, key: CacheKey, suggested_path: Path) -> None:
        self.key = key
        self.suggested_path = suggested_path
        super().__init__(
            f"No replay fixture for cache key {key.to_sha256()}. "
            f"Run with {RECORD_ENV_VAR}=1 to record one at {suggested_path}."
        )


class ReplayStore:
    """JSON fixture store keyed by :meth:`CacheKey.to_sha256`.

    Each fixture is a single JSON document containing the recorded value plus
    the ``CacheKey`` components for human readability. Only the value is
    returned to callers via :meth:`get`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._directory = Path(self._settings.replay_fixtures_dir).expanduser()

    @property
    def directory(self) -> Path:
        return self._directory

    def path_for(self, key: CacheKey) -> Path:
        """Return the on-disk path for ``key``'s fixture."""

        return self._directory / f"{key.to_sha256()}.json"

    def get(self, key: CacheKey) -> Any:
        """Return the recorded value for ``key`` or raise :class:`ReplayMissError`."""

        path = self.path_for(key)
        if not path.is_file():
            raise ReplayMissError(key, path)
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayMissError(key, path) from exc
        if not isinstance(payload, dict) or "value" not in payload:
            raise ReplayMissError(key, path)
        return payload["value"]

    def record(self, key: CacheKey, value: Any) -> Path:
        """Write ``value`` as a fixture for ``key``.

        No-op (returns the path it *would* have written) when
        :func:`is_recording` is ``False``. Callers can rely on the returned
        path for logging.
        """

        path = self.path_for(key)
        if not is_recording():
            return path
        self._directory.mkdir(parents=True, exist_ok=True)
        payload = {"key": key.model_dump(), "value": value}
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
        return path

    def has(self, key: CacheKey) -> bool:
        """Return ``True`` if a fixture exists for ``key``."""

        return self.path_for(key).is_file()
