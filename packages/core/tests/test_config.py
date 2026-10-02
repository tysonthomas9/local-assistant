from pathlib import Path

import pytest
from pydantic import ValidationError

from assistant_core.config import (
    AssistantConfig,
    ConfigError,
    LlmConfig,
    deep_merge,
    load_config,
    parse_cli_overrides,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    (tmp_path / "profiles").mkdir()
    (tmp_path / "assistant.toml").write_text(
        '[llm]\nbase_url = "http://base/v1"\nmodel = "base-model"\nmax_concurrency = 6\n'
        '[net]\nedgelink_bind = "0.0.0.0:8770"\n'
    )
    (tmp_path / "profiles" / "p.toml").write_text(
        '[llm]\nbase_url = "http://profile/v1"\n[edge]\nbody = "reachy"\n'
    )
    return tmp_path


def test_code_defaults_without_files(tmp_path: Path) -> None:
    config = load_config(tmp_path, environ={})
    assert config == AssistantConfig()
    assert config.llm.max_concurrency == 8
    assert config.llm.reserved_voice_slots == 1
    assert config.privacy.online_skills is False
    assert config.net.tls_cert is None


def test_precedence_defaults_file_profile_env_cli(config_dir: Path) -> None:
    base = load_config(config_dir, environ={})
    assert (base.llm.impl, base.llm.base_url, base.llm.model) == (
        "openai",
        "http://base/v1",
        "base-model",
    )
    assert base.llm.ctx == 16384  # untouched default

    profiled = load_config(config_dir, profile="p", environ={})
    assert profiled.llm.base_url == "http://profile/v1"
    assert profiled.edge.body == "reachy"
    assert profiled.llm.model == "base-model"  # tables deep-merge
    assert profiled.net.edgelink_bind == "0.0.0.0:8770"

    env = {"ASSISTANT__LLM__BASE_URL": "http://env/v1", "ASSISTANT__LLM__MAX_CONCURRENCY": "3"}
    from_env = load_config(config_dir, profile="p", environ=env)
    assert from_env.llm.base_url == "http://env/v1"
    assert from_env.llm.max_concurrency == 3

    cli = load_config(
        config_dir,
        profile="p",
        environ=env,
        cli_overrides=parse_cli_overrides(["llm.base_url=http://cli/v1"]),
    )
    assert cli.llm.base_url == "http://cli/v1"
    assert cli.llm.max_concurrency == 3


def test_env_is_read_from_os_environ_by_default(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSISTANT__LLM__BASE_URL", "http://os-env/v1")
    monkeypatch.setenv("assistant__privacy__online_skills", "true")
    config = load_config(config_dir)
    assert config.llm.base_url == "http://os-env/v1"
    assert config.privacy.online_skills is True


@pytest.mark.parametrize(
    "llm",
    [
        {"max_concurrency": 0, "reserved_voice_slots": 0},
        {"max_concurrency": 1, "reserved_voice_slots": 1},
        {"max_concurrency": 4, "reserved_voice_slots": 4},
        {"reserved_voice_slots": -1},
        {"impl": "ollama"},
        {"impl": "fake"},
        {"server": "llama-server"},
    ],
)
def test_llm_validation(llm: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LlmConfig.model_validate(llm)


def test_llm_validation_through_env(config_dir: Path) -> None:
    with pytest.raises(ValidationError, match="reserved_voice_slots"):
        load_config(config_dir, environ={"ASSISTANT__LLM__RESERVED_VOICE_SLOTS": "6"})
    ok = load_config(
        config_dir,
        environ={
            "ASSISTANT__LLM__MAX_CONCURRENCY": "1",
            "ASSISTANT__LLM__RESERVED_VOICE_SLOTS": "0",
        },
    )
    assert ok.llm.max_concurrency == 1


def test_unknown_keys_are_refused(tmp_path: Path) -> None:
    (tmp_path / "assistant.toml").write_text('[llm]\nbase_ulr = "typo"\n')
    with pytest.raises(ValidationError):
        load_config(tmp_path, environ={})


def test_tls_cert_and_key_go_together(tmp_path: Path) -> None:
    (tmp_path / "assistant.toml").write_text('[net]\ntls_cert = "brain.crt"\n')
    with pytest.raises(ValidationError):
        load_config(tmp_path, environ={})
    (tmp_path / "assistant.toml").write_text('[net]\ntls_cert = "a.crt"\ntls_key = "a.key"\n')
    assert load_config(tmp_path, environ={}).net.tls_key == Path("a.key")


def test_missing_profile_and_bad_toml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path, profile="nope", environ={})
    (tmp_path / "assistant.toml").write_text("[llm\n")
    with pytest.raises(ConfigError):
        load_config(tmp_path, environ={})


def test_cli_override_parsing() -> None:
    assert parse_cli_overrides(
        ["llm.max_concurrency=4", "llm.model=qwen3", "brain.announce_devices=['a']"]
    ) == {
        "llm": {"max_concurrency": 4, "model": "qwen3"},
        "brain": {"announce_devices": ["a"]},
    }
    with pytest.raises(ConfigError):
        parse_cli_overrides(["no-equals"])


def test_deep_merge() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}, "d": 1}, {"a": {"b": 3}, "d": {"x": 1}}) == {
        "a": {"b": 3, "c": 2},
        "d": {"x": 1},
    }


def test_data_dir_is_expanded() -> None:
    assert "~" not in str(AssistantConfig().brain.data_dir)
