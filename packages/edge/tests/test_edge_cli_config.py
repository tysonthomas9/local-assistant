"""The edge reads its config: the base config runs as is, flags override it."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from assistant_core.config import AssistantConfig, load_config
from assistant_edge.agent import MIC_CAP_S
from assistant_edge.cli import build, make_parser

pytestmark = pytest.mark.unit

CONFIG = Path(__file__).parents[3] / "config"


def test_base_config_runs_as_is() -> None:
    config = load_config(CONFIG, environ={})
    kind, body, options = build(make_parser().parse_args([]), config)
    assert kind == "reachy"
    assert body == {
        "connection": "localhost_only",
        "expressions": "config/bodies/reachy.toml",
        "idle_sleep_s": 120,
        "tracking": "voice+face",
    }
    assert options.device_id == "lite"
    assert options.url == "ws://127.0.0.1:8770/edge/v1"  # brain_url "mdns": this machine
    assert options.listen == "wake_word"
    assert options.aec == "hw"
    assert options.max_window_s == MIC_CAP_S == 120
    assert options.wake_no_speech_s == 8
    assert options.barge_in_stop_delay_ms == 350
    assert (CONFIG.parent / body["expressions"]).is_file()


def test_flags_override_the_config() -> None:
    config = load_config(CONFIG, profile="ci", environ={})
    args = make_parser().parse_args(
        [
            *("--device-id", "desk", "--body", "console", "--url", "ws://127.0.0.1:9/edge/v1"),
            *("--listen", "open_mic", "--wake-word", "hey_marvin"),
        ]
    )
    kind, body, options = build(args, config)
    assert (kind, body) == ("console", {})
    assert (options.device_id, options.url) == ("desk", "ws://127.0.0.1:9/edge/v1")
    assert (options.listen, options.wake_words) == ("open_mic", ("hey_marvin",))
    assert options.aec == "none"  # the ci profile's console body


def test_a_body_option_overrides_the_body_table() -> None:
    config = load_config(CONFIG, environ={})
    args = make_parser().parse_args(
        ["--body-option", "idle_sleep_s=8", "--body-option", "connection=network"]
    )
    _, body, _ = build(args, config)
    assert body["idle_sleep_s"] == 8
    assert body["connection"] == "network"


def test_the_mic_cap_is_two_minutes_at_most() -> None:
    with pytest.raises(ValidationError):
        AssistantConfig.model_validate({"edge": {"wake": {"max_window_s": 121}}})
