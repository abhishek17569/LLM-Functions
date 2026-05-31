"""Five-layer precedence and override-logging tests for the config resolver."""

from __future__ import annotations

import logging

import pytest

from llm_functions._config import (
    ConfigurationError,
    ProviderConfig,
    Settings,
    configure,
    reset_settings,
    resolve,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_settings()


def test_defaults_alone() -> None:
    s = resolve()
    assert s == Settings()


def test_global_overrides_defaults() -> None:
    configure(model="anthropic/claude-3-5-haiku", temperature=0.4)
    s = resolve()
    assert s.model == "anthropic/claude-3-5-haiku"
    assert s.temperature == 0.4


def test_decorator_overrides_global() -> None:
    configure(model="anthropic/claude-3-5-haiku")
    s = resolve(decorator={"model": "openai/gpt-4o"})
    assert s.model == "openai/gpt-4o"


def test_docstring_overrides_decorator() -> None:
    s = resolve(
        decorator={"model": "openai/gpt-4o"},
        docstring={"model": "anthropic/claude-3-5-sonnet"},
    )
    assert s.model == "anthropic/claude-3-5-sonnet"


def test_call_overrides_everything() -> None:
    configure(model="anthropic/claude-3-5-haiku")
    s = resolve(
        decorator={"model": "openai/gpt-4o"},
        docstring={"model": "anthropic/claude-3-5-sonnet"},
        call={"model": "openai/gpt-4o-mini"},
    )
    assert s.model == "openai/gpt-4o-mini"


def test_field_by_field_merge_does_not_drop_other_layers() -> None:
    configure(temperature=0.7)
    s = resolve(
        decorator={"max_repairs": 5},
        docstring={"model": "openai/gpt-4o"},
        call={"timeout": 12.0},
    )
    assert s.temperature == 0.7
    assert s.max_repairs == 5
    assert s.model == "openai/gpt-4o"
    assert s.timeout == 12.0


def test_docstring_override_logs_when_global_is_non_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    configure(model="anthropic/claude-3-5-haiku")
    with caplog.at_level(logging.INFO, logger="llm_functions"):
        s = resolve(docstring={"model": "openai/gpt-4o-mini"})
    assert s.model == "openai/gpt-4o-mini"
    messages = [r.getMessage() for r in caplog.records if r.name == "llm_functions"]
    assert any("docstring overrides global model" in m for m in messages)


def test_docstring_override_silent_when_global_is_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="llm_functions"):
        resolve(docstring={"model": "anthropic/claude-3-5-sonnet"})
    messages = [r.getMessage() for r in caplog.records if r.name == "llm_functions"]
    assert not any("docstring overrides global" in m for m in messages)


def test_docstring_override_silent_when_value_matches_global(
    caplog: pytest.LogCaptureFixture,
) -> None:
    configure(cache="off")
    with caplog.at_level(logging.INFO, logger="llm_functions"):
        resolve(docstring={"cache": "off"})
    messages = [r.getMessage() for r in caplog.records if r.name == "llm_functions"]
    assert not any("docstring overrides global" in m for m in messages)


def test_unknown_configure_key_raises() -> None:
    with pytest.raises(ConfigurationError):
        configure(does_not_exist=1)


def test_unknown_layer_key_raises() -> None:
    with pytest.raises(ConfigurationError):
        resolve(decorator={"does_not_exist": 1})


def test_invalid_value_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        configure(temperature=-1.0)
    with pytest.raises(ConfigurationError):
        resolve(call={"cache": "maybe"})  # type: ignore[arg-type]


def test_none_values_in_layer_are_ignored() -> None:
    configure(model="anthropic/claude-3-5-haiku")
    s = resolve(decorator={"model": None}, docstring={"cache": "replay"})
    assert s.model == "anthropic/claude-3-5-haiku"
    assert s.cache == "replay"


def test_configure_providers_with_provider_config() -> None:
    configure(
        providers={
            "openai": ProviderConfig(api_key="sk-O"),
            "anthropic": ProviderConfig(api_key="sk-A"),
        }
    )
    s = resolve()
    assert s.providers is not None
    assert s.providers["openai"].api_key == "sk-O"
    assert s.providers["anthropic"].api_key == "sk-A"


def test_configure_providers_accepts_dict_form() -> None:
    configure(
        providers={
            "openai": {
                "api_key": "sk-O",
                "api_base": "https://gateway.example.com/v1",
                "extra_headers": {"X-Tenant": "foo"},
            },
        }
    )
    s = resolve()
    assert s.providers is not None
    cfg = s.providers["openai"]
    assert cfg.api_key == "sk-O"
    assert cfg.api_base == "https://gateway.example.com/v1"
    assert cfg.extra_headers == {"X-Tenant": "foo"}


def test_configure_providers_merges_across_calls() -> None:
    configure(providers={"openai": ProviderConfig(api_key="sk-O")})
    configure(providers={"anthropic": ProviderConfig(api_key="sk-A")})
    s = resolve()
    assert s.providers is not None
    assert set(s.providers.keys()) == {"openai", "anthropic"}


def test_configure_providers_overwrites_same_provider() -> None:
    configure(providers={"openai": ProviderConfig(api_key="sk-old")})
    configure(providers={"openai": ProviderConfig(api_key="sk-new")})
    s = resolve()
    assert s.providers is not None
    assert s.providers["openai"].api_key == "sk-new"


def test_configure_providers_rejects_non_dict() -> None:
    with pytest.raises(ConfigurationError):
        configure(providers="sk-O")  # type: ignore[arg-type]


def test_configure_providers_rejects_unknown_provider_field() -> None:
    with pytest.raises(ConfigurationError):
        configure(providers={"openai": {"api_key": "sk-O", "bogus": 1}})


def test_provider_config_callable_api_key_resolves_per_call() -> None:
    counter = {"n": 0}

    def fresh_jwt() -> str:
        counter["n"] += 1
        return f"jwt-{counter['n']}"

    cfg = ProviderConfig(api_key=fresh_jwt)
    assert cfg.resolve_api_key() == "jwt-1"
    assert cfg.resolve_api_key() == "jwt-2"


def test_provider_config_callable_api_key_must_return_str() -> None:
    cfg = ProviderConfig(api_key=lambda: 42)  # type: ignore[arg-type,return-value]
    with pytest.raises(ConfigurationError):
        cfg.resolve_api_key()


def test_provider_config_resolve_returns_none_when_unset() -> None:
    assert ProviderConfig().resolve_api_key() is None


def test_configure_log_level_attaches_handler_and_disables_propagation() -> None:
    import io

    buf = io.StringIO()
    configure(log_level="DEBUG", log_stream=buf)
    logger = logging.getLogger("llm_functions")
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    logger.debug("hello-from-test")
    output = buf.getvalue()
    assert "hello-from-test" in output
    assert "llm_function" in output


def test_configure_log_level_is_idempotent_no_double_handlers() -> None:
    import io

    buf1 = io.StringIO()
    buf2 = io.StringIO()
    configure(log_level="INFO", log_stream=buf1)
    configure(log_level="DEBUG", log_stream=buf2)
    logger = logging.getLogger("llm_functions")
    owned = [h for h in logger.handlers if getattr(h, "_llm_functions_owned", False)]
    assert len(owned) == 1
    logger.debug("only-buf2")
    assert "only-buf2" not in buf1.getvalue()
    assert "only-buf2" in buf2.getvalue()


def test_configure_log_level_int_accepted() -> None:
    import io

    buf = io.StringIO()
    configure(log_level=logging.WARNING, log_stream=buf)
    logger = logging.getLogger("llm_functions")
    assert logger.level == logging.WARNING


def test_configure_invalid_log_level_raises() -> None:
    with pytest.raises(ConfigurationError):
        configure(log_level=object())  # type: ignore[arg-type]
