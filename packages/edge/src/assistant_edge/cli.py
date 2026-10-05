"""`python -m assistant_edge` / `assistant-edge`: run the edge agent with one body.

It reads `assistant.toml` (`--config-dir`, plus `--profile`): `[edge]` (device id, body, brain
URL), `[edge.audio]`, `[edge.listen]`, `[edge.vad]`, `[edge.wake]` and the body's own table
(`[body.reachy]`, ...). Command-line flags override it; the base config runs as is (until mDNS
discovery exists, `brain_url = "mdns"` means the brain on this machine).
"""

import argparse
import asyncio
import os
import tomllib
from pathlib import Path
from typing import Any

from assistant_core.config import AssistantConfig, ConfigError, load_config
from assistant_edge.agent import LISTEN_MODES, AgentOptions, EdgeAgent
from assistant_edge.bodies import available, load_body
from assistant_link.console import DEFAULT_DEV_TOKEN, DEV_TOKEN_ENV
from assistant_link.server import DEFAULT_PORT, EDGE_PATH


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config_dir, profile=args.profile)
        kind, body_options, options = build(args, config)
        body = load_body(kind, **body_options)
    except (ConfigError, ValueError, TypeError) as exc:
        parser.error(f"config: {exc}")
    return asyncio.run(EdgeAgent(body, options).run())


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m assistant_edge")
    parser.add_argument("--device-id", default=None, help="default: [edge] device_id")
    parser.add_argument(
        "--body", default=None, help=f"one of: {', '.join(available())} (default: [edge] body)"
    )
    parser.add_argument("--url", default=None, help="default: [edge] brain_url")
    parser.add_argument("--token", default=os.environ.get(DEV_TOKEN_ENV, DEFAULT_DEV_TOKEN))
    parser.add_argument(
        "--energy-trigger-dbfs",
        type=float,
        default=None,
        help="open a mic window when the level stays above this many dBFS (default: off)",
    )
    parser.add_argument(
        "--no-smart-turn",
        action="store_true",
        help="end turns on the speech detector's silence alone ([engine] smart_turn off)",
    )
    parser.add_argument(
        "--vad-end-ms",
        type=int,
        default=None,
        help="close a mic window after this much quiet once speech was heard (energy trigger: off "
        "by default; speech detector: [edge.vad] end_ms)",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="also write each played speech stream to <dir>/stream-<id>.wav",
    )
    parser.add_argument(
        "--energy-trigger-feed-only",
        action="store_true",
        help="tests: the energy trigger fires only on audio fed with /feed, not on room sound",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="read [edge], [edge.*] and [body.<body>] from <dir>/assistant.toml",
    )
    parser.add_argument("--profile", default=None, help="overlay <dir>/profiles/<profile>.toml")
    parser.add_argument(
        "--listen",
        choices=LISTEN_MODES,
        default=None,
        help="wake_word, open_mic or push_to_talk (default: [edge.listen] mode, wake_word)",
    )
    parser.add_argument("--wake-word", action="append", default=None, help="a wake-word model")
    parser.add_argument("--wake-threshold", type=float, default=None)
    parser.add_argument(
        "--body-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="set one [body.<body>] option over the config (a TOML value, e.g. idle_sleep_s=8)",
    )
    return parser


def _body_option(text: str) -> tuple[str, Any]:
    key, sep, value = text.partition("=")
    if not sep or not key.strip():
        raise SystemExit(f"--body-option {text!r}: want KEY=VALUE")
    try:
        parsed = tomllib.loads(f"v = {value}")["v"]
    except tomllib.TOMLDecodeError:
        parsed = value  # a bare string
    return key.strip(), parsed


def brain_url(configured: str) -> str:
    """`[edge] brain_url`: a static URL, or `mdns` (the brain on this machine until the
    discovery exists)."""
    if configured == "mdns":
        return f"ws://127.0.0.1:{DEFAULT_PORT}{EDGE_PATH}"
    return configured


def body_options(kind: str, config: AssistantConfig) -> dict[str, Any]:
    """The body's own table (`[body.<kind>]`); the sounddevice body also takes the
    `[edge.audio]` devices."""
    options = dict(config.body.get(kind, {}))
    if kind == "sounddevice":
        options.setdefault("input", config.edge.audio.input)
        options.setdefault("output", config.edge.audio.output)
    return options


def build(
    args: argparse.Namespace, config: AssistantConfig
) -> tuple[str, dict[str, Any], AgentOptions]:
    """The body kind, its options and the agent options: the flags over the config."""
    edge = config.edge
    kind = args.body or edge.body
    options = AgentOptions(
        device_id=args.device_id or edge.device_id,
        url=args.url or brain_url(edge.brain_url),
        token=args.token,
        aec=edge.audio.aec if kind == edge.body else None,  # [edge.audio] is the config's body's
        energy_trigger_dbfs=args.energy_trigger_dbfs,
        record_dir=args.record_dir,
        energy_trigger_feed_only=args.energy_trigger_feed_only,
        listen=args.listen or edge.listen.mode,
        max_window_s=edge.wake.max_window_s,
        wake_engine=edge.wake.engine,
        wake_words=tuple(args.wake_word or edge.wake.models),
        wake_threshold=args.wake_threshold or edge.wake.threshold,
        wake_pre_roll_ms=int(edge.wake.pre_roll_s * 1000),
        wake_no_speech_s=edge.wake.no_speech_s,
        phrase_model_dir=Path(edge.wake.phrase_model_dir) if edge.wake.phrase_model_dir else None,
        vad_threshold=edge.vad.threshold,
        vad_start_ms=edge.vad.start_ms,
        vad_end_ms=args.vad_end_ms,
        speech_end_ms=args.vad_end_ms or edge.vad.end_ms,
        vad_pre_roll_ms=edge.vad.pre_roll_ms,
        barge_in_stop_delay_ms=edge.vad.barge_in_stop_delay_ms,
        smart_turn=config.engine.smart_turn and not args.no_smart_turn,
        smart_turn_threshold=config.engine.smart_turn_threshold,
        smart_turn_max_wait_ms=config.engine.smart_turn_max_wait_ms,
    )
    body = body_options(kind, config)
    body.update(_body_option(text) for text in args.body_option)
    return kind, body, options
