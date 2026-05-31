"""Record JSON replay fixtures for the e2e tests.

Run from the project root:

    uv run python scripts/record_e2e_fixtures.py

Writes one ``<sha256>.json`` per (function, call) pair into
``tests/fixtures/llmfunctionkit/``. Run again any time the function source,
docstring, or call args change so the cache key stays in sync.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def main() -> None:
    from llmfunctionkit._cache import CacheKey
    from llmfunctionkit._config import configure, reset_settings
    from llmfunctionkit._decorator import _build_cache_key_for_call, _set_provider_factory
    from llmfunctionkit._provider import _OUTPUT_TOOL_NAME, Provider

    reset_settings()
    fixtures_dir = ROOT / "tests" / "fixtures" / "llmfunctionkit"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    configure(replay_fixtures_dir=str(fixtures_dir))

    # Stub-out litellm.supports_function_calling so tests behave the same as
    # at runtime under our fake completion.
    import litellm

    litellm.supports_function_calling = lambda model: True  # type: ignore[assignment]

    # Import the e2e test module so its decorated functions are constructed
    # against the configured replay dir.
    import test_e2e  # type: ignore[import-not-found]

    targets: list[tuple[Any, tuple[Any, ...], dict[str, Any], Any]] = [
        (test_e2e.is_man, ("a fellow named John walked in",), {}, {"value": True}),
        (test_e2e.find_names, ("Alice and Bob went home.",), {}, {"value": ["Alice", "Bob"]}),
        # find_names raise example — the LLM signals the declared exception
        # by invoking the synthesised raise_NameNotFoundError tool. We record
        # the resulting "tool_calls" message; on replay the decorator catches
        # _LLMFunctionRaisedException at the boundary and re-raises.
        # For replay purposes, the raise path can't be cached as a value, so
        # we record a normal value and adjust the e2e test to assert on the
        # exception path with cache="off".
    ]

    for fn, args, kwargs, fake_value in targets:
        key: CacheKey = _build_cache_key_for_call(fn, args, kwargs)
        path = fixtures_dir / f"{key.to_sha256()}.json"
        payload = {"key": key.model_dump(), "value": fake_value}
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"wrote {path}")

    # Streaming fixture for greet — the cached value is the assembled string.
    greet_key = _build_cache_key_for_call(test_e2e.greet, ("Alice",), {})
    greet_path = fixtures_dir / f"{greet_key.to_sha256()}.json"
    payload = {"key": greet_key.model_dump(), "value": "Hello, Alice!"}
    with greet_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {greet_path}")

    # Avoid an unused-import warning at the top.
    _ = Provider, _OUTPUT_TOOL_NAME, _set_provider_factory


if __name__ == "__main__":
    main()
