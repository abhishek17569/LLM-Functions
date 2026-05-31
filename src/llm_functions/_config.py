"""Settings and resolver for llm_function.

The resolver merges five precedence layers (defaults < global < decorator <
docstring < call) field-by-field. When a docstring directive overrides a
non-default global ``model`` / ``cache``, the resolver emits an INFO log so
that callers can audit per-function deviations from a configured default.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ConfigurationError",
    "LLMFunctionError",
    "ProviderConfig",
    "Settings",
    "configure",
    "get_settings",
    "reset_settings",
    "resolve",
]


CacheMode = Literal["on", "off", "replay"]
ApiKeySource = str | Callable[[], str]


class LLMFunctionError(Exception):
    """Base class for all llm_function errors.

    Stub-defined here so the foundational layer has no module dependencies;
    forge-decorator re-exports it from ``_exceptions.py`` later.
    """


class ConfigurationError(LLMFunctionError):
    """Raised when configuration is invalid (bad cache mode, unknown setting, etc.)."""


_LOGGER = logging.getLogger("llm_functions")


@dataclass(frozen=True)
class ProviderConfig:
    """Per-provider credentials and gateway configuration.

    ``api_key`` may be a string or a zero-argument callable; the callable is
    invoked once per LLM call so JWT-refreshable tokens (TrueFoundry, Vertex,
    custom OAuth gateways) can rotate without reconfiguring the library.
    ``api_base`` overrides the upstream URL; ``extra_headers`` are forwarded
    verbatim and are useful for tenant/project metadata required by gateways.
    """

    api_key: ApiKeySource | None = None
    api_base: str | None = None
    extra_headers: dict[str, str] | None = field(default=None)

    def resolve_api_key(self) -> str | None:
        """Return the live API key, calling the callable form if present."""

        if self.api_key is None:
            return None
        if callable(self.api_key):
            value = self.api_key()
            if not isinstance(value, str):
                raise ConfigurationError(
                    f"api_key callable must return str, got {type(value).__name__}"
                )
            return value
        return self.api_key


class Settings(BaseModel):
    """Runtime settings for llm_function calls.

    All fields are merged field-by-field through :func:`resolve`.
    """

    model_config = ConfigDict(extra="forbid", frozen=False, arbitrary_types_allowed=True)

    model: str = "openai/gpt-4o-mini"
    cache: CacheMode = "on"
    cache_dir: str | None = None
    temperature: float = Field(default=0.0, ge=0.0)
    max_repairs: int = Field(default=2, ge=0)
    max_tool_iterations: int = Field(default=10, ge=0)
    timeout: float = Field(default=60.0, gt=0.0)
    allow_all_tools: bool = False
    replay_fixtures_dir: str = "tests/fixtures/llm_functions"
    providers: dict[str, ProviderConfig] | None = Field(
        default=None,
        description=(
            "Per-provider configuration keyed by LiteLLM provider prefix "
            "(``openai``, ``anthropic``, …). Each entry carries the api_key "
            "(string or refresh-callable), an optional api_base URL for "
            "OpenAI-compatible gateways like TrueFoundry, and optional "
            "extra_headers (tenant/project metadata). Providers without an "
            "entry fall back to environment variables."
        ),
    )


_DEFAULT_SETTINGS: Settings = Settings()
_GLOBAL_SETTINGS: Settings = Settings()

_UNSET: Any = object()
_LLM_FUNCTIONS_HANDLER_TAG = "_llm_functions_owned"


def _install_logger(*, level: int | str, stream: Any) -> None:
    """Attach a single dedicated handler to the ``llm_function`` logger.

    Idempotent — calling ``configure(log_level=...)`` twice swaps the handler
    rather than stacking. ``propagate`` is set to ``False`` so DEBUG records
    don't bubble up to a noisy root logger that would also print LiteLLM,
    httpx, urllib3, etc.
    """

    import sys

    target_level = logging.getLevelName(level) if isinstance(level, str) else level
    if not isinstance(target_level, int):
        raise ConfigurationError(
            f"log_level must be a level name (e.g. 'DEBUG') or int, got {level!r}"
        )

    logger = logging.getLogger("llm_functions")
    logger.setLevel(target_level)
    logger.propagate = False

    # Remove any handler we previously installed so repeated configure() calls
    # don't double-print.
    for existing in list(logger.handlers):
        if getattr(existing, _LLM_FUNCTIONS_HANDLER_TAG, False):
            logger.removeHandler(existing)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    setattr(handler, _LLM_FUNCTIONS_HANDLER_TAG, True)
    logger.addHandler(handler)


def _allowed_field_names() -> set[str]:
    return set(Settings.model_fields.keys())


def configure(**kwargs: Any) -> None:
    """Update the process-global :class:`Settings`.

    Unknown keys raise :class:`ConfigurationError`. The update is partial:
    fields not supplied keep their current global value.

    The non-``Settings`` kwargs ``log_level`` and ``log_stream`` install a
    dedicated handler on the ``llm_function`` logger and disable propagation
    to the root logger — so DEBUG output from this library doesn't pull in
    LiteLLM, urllib3, etc.
    """

    global _GLOBAL_SETTINGS

    log_level = kwargs.pop("log_level", _UNSET)
    log_stream = kwargs.pop("log_stream", _UNSET)
    if log_level is not _UNSET or log_stream is not _UNSET:
        _install_logger(
            level=log_level if log_level is not _UNSET else "DEBUG",
            stream=log_stream if log_stream is not _UNSET else None,
        )

    allowed = _allowed_field_names()
    unknown = set(kwargs) - allowed
    if unknown:
        raise ConfigurationError(
            f"Unknown configuration key(s): {sorted(unknown)}. Allowed: {sorted(allowed)}"
        )
    if "providers" in kwargs and kwargs["providers"] is not None:
        # Merge per-provider entries so a follow-up ``configure`` call adding
        # one provider does not wipe previously configured ones. Pass
        # ``providers=None`` explicitly to clear all providers.
        existing: dict[str, ProviderConfig] = dict(_GLOBAL_SETTINGS.providers or {})
        existing.update(_coerce_providers(kwargs["providers"]))
        kwargs = {**kwargs, "providers": existing}

    try:
        _GLOBAL_SETTINGS = _GLOBAL_SETTINGS.model_copy(update=kwargs)
        # Validate scalar fields by re-instantiating; preserve providers since
        # the dataclass round-trips poorly through model_dump.
        scalars = {k: v for k, v in _GLOBAL_SETTINGS.model_dump().items() if k != "providers"}
        validated = Settings(**scalars)
        validated.providers = _GLOBAL_SETTINGS.providers
        _GLOBAL_SETTINGS = validated
    except Exception as exc:  # pydantic ValidationError or similar
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc


def get_settings() -> Settings:
    """Return the current process-global :class:`Settings`."""

    return _GLOBAL_SETTINGS


def reset_settings() -> None:
    """Reset the process-global settings to library defaults. Test-only helper.

    Also removes any handler installed by ``configure(log_level=...)`` and
    restores the ``llm_function`` logger to ``propagate=True``, so cross-test
    bleed doesn't break ``caplog``-based assertions in unrelated tests.
    """

    global _GLOBAL_SETTINGS
    _GLOBAL_SETTINGS = Settings()
    logger = logging.getLogger("llm_functions")
    for existing in list(logger.handlers):
        if getattr(existing, _LLM_FUNCTIONS_HANDLER_TAG, False):
            logger.removeHandler(existing)
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


def _coerce_providers(value: Any) -> dict[str, ProviderConfig]:
    """Coerce dict-of-dict input into ``dict[str, ProviderConfig]``.

    Accepts an existing ``dict[str, ProviderConfig]`` unchanged so callers can
    pass either form; rejects anything else with ``ConfigurationError``.
    """

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(
            f"providers must be a dict[str, ProviderConfig | dict], got {type(value).__name__}"
        )
    out: dict[str, ProviderConfig] = {}
    for key, entry in value.items():
        if not isinstance(key, str):
            raise ConfigurationError(f"provider keys must be str, got {type(key).__name__}")
        if isinstance(entry, ProviderConfig):
            out[key] = entry
            continue
        if isinstance(entry, dict):
            allowed = {"api_key", "api_base", "extra_headers"}
            unknown = set(entry) - allowed
            if unknown:
                raise ConfigurationError(
                    f"unknown provider field(s) for {key!r}: {sorted(unknown)}; "
                    f"allowed: {sorted(allowed)}"
                )
            out[key] = ProviderConfig(**entry)
            continue
        raise ConfigurationError(
            f"provider entry for {key!r} must be ProviderConfig or dict, got {type(entry).__name__}"
        )
    return out


def _filter_known(layer: dict[str, Any] | None, *, layer_name: str) -> dict[str, Any]:
    if not layer:
        return {}
    allowed = _allowed_field_names()
    unknown = set(layer) - allowed
    if unknown:
        raise ConfigurationError(
            f"Unknown {layer_name} setting(s): {sorted(unknown)}. Allowed: {sorted(allowed)}"
        )
    return {k: v for k, v in layer.items() if v is not None}


def resolve(
    *,
    decorator: dict[str, Any] | None = None,
    docstring: dict[str, Any] | None = None,
    call: dict[str, Any] | None = None,
) -> Settings:
    """Merge global settings with the three caller-supplied layers.

    Precedence (lowest to highest): library defaults, global settings,
    decorator kwargs, docstring directives (``Model:`` / ``Cache:``), call-site
    overrides. ``None`` values in a layer are ignored so omitted fields fall
    through to the layer below.

    Logs an INFO message via ``logging.getLogger("llm_functions")`` when a
    docstring directive overrides a non-default global ``model`` or ``cache``
    value.
    """

    decorator_layer = _filter_known(decorator, layer_name="decorator")
    docstring_layer = _filter_known(docstring, layer_name="docstring")
    call_layer = _filter_known(call, layer_name="call")

    merged: dict[str, Any] = _GLOBAL_SETTINGS.model_dump()
    merged.update(decorator_layer)

    for setting_name in ("model", "cache"):
        if setting_name not in docstring_layer:
            continue
        global_value = getattr(_GLOBAL_SETTINGS, setting_name)
        default_value = getattr(_DEFAULT_SETTINGS, setting_name)
        new_value = docstring_layer[setting_name]
        if global_value != default_value and global_value != new_value:
            _LOGGER.info(
                "llm_functions: docstring overrides global %s=%r with %r",
                setting_name,
                global_value,
                new_value,
            )

    merged.update(docstring_layer)
    merged.update(call_layer)

    try:
        return Settings(**merged)
    except Exception as exc:
        raise ConfigurationError(f"Invalid resolved settings: {exc}") from exc
