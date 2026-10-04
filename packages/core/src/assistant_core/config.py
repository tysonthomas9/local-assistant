"""Layered TOML configuration (proposal section 5.5).

Precedence, lowest first:

    code defaults < config/assistant.toml < config/profiles/<profile>.toml
                  < environment ASSISTANT__SECTION__KEY < CLI overrides

Tables are deep-merged, so a profile only lists what it changes. Environment variables use
`__` as the nesting separator: `ASSISTANT__LLM__BASE_URL=http://...` sets `[llm] base_url`.
"""

import os
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict

ENV_PREFIX = "ASSISTANT__"
ENV_DELIMITER = "__"


class ConfigError(ValueError):
    """A config file is missing or invalid."""


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- sections


class NetConfig(Section):
    edgelink_bind: str = "127.0.0.1:8770"
    """The only LAN-facing listener. Loopback unless a profile opens it."""
    admin_bind: str = "127.0.0.1:8771"
    tls_cert: Path | None = None
    tls_key: Path | None = None

    @field_validator("tls_cert", "tls_key", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def _cert_and_key_together(self) -> Self:
        if (self.tls_cert is None) != (self.tls_key is None):
            raise ValueError("net.tls_cert and net.tls_key must be set together")
        return self


class PrivacyConfig(Section):
    online_skills: bool = False
    hard_mute_button: bool = True


class BrainConfig(Section):
    data_dir: Path = Field(default=Path("~/.local/share/assistant"), validate_default=True)
    default_assistant: str = "jarvis"
    wake_arbitration_ms: int = Field(default=200, ge=0)
    follow_up_s: float = Field(default=10, ge=0)
    announce_devices: list[str] = Field(default_factory=list)

    @field_validator("data_dir", mode="after")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return value.expanduser()


class EngineConfig(Section):
    impl: Literal["basic", "realtime"] = "basic"
    vad_stop_s: float = Field(default=0.2, gt=0)
    smart_turn: bool = True
    """Smart Turn v3.2 (pipecat-ai/smart-turn-v3) decides on the edge whether a pause ends the
    turn: when the speech detector hears `[edge.vad] end_ms` of quiet, the model looks at the
    turn's audio; a turn that sounds unfinished stays open until `smart_turn_max_wait_ms` of
    quiet (speech resuming meanwhile goes on in the same turn)."""
    smart_turn_threshold: float = Field(default=0.5, gt=0, lt=1)
    smart_turn_max_wait_ms: int = Field(default=2000, ge=100, le=5000)
    realtime_url: str | None = None
    """HF speech-to-speech (OpenAI Realtime) endpoint for the `realtime` engine (phase 2)."""
    speech: bool = True
    """The basic engine speaks: speech input through `[stt]`, replies as audio through `[tts]`
    (the speech server). Off: replies are speak text and speech input is not understood."""


class LlmPriority(Section):
    """vLLM request priority per class (lower runs first)."""

    voice: int = 0
    proactive: int = 5
    background: int = 10


class LlmConfig(Section):
    """Any OpenAI-compatible server: vLLM by default, Ollama or llama-server as fallback."""

    impl: Literal["openai"] = "openai"
    """OpenAI-compatible HTTP; the only client there is."""
    server: Literal["vllm", "ollama"] = "vllm"
    """The stack's LLM server (`scripts/llm_server.sh`): vLLM by default, Ollama the fallback.
    Both serve the same model name on the same `base_url`; only vLLM gets `priority`."""
    base_url: str = "http://127.0.0.1:8773/v1"
    model: str = "gemma4-26b"
    ctx: int = Field(default=16384, gt=0)
    max_concurrency: int = Field(default=8, ge=1)
    """Requests in flight at once, across all sessions (the priority gate)."""
    reserved_voice_slots: int = Field(default=1, ge=0)
    """Slots only voice requests may use; must leave at least one for everyone else."""
    send_priority: bool = True
    """Send `priority` with each request when `server` is vllm (it runs
    `--scheduling-policy priority`); Ollama has no request priority."""
    priority: LlmPriority = Field(default_factory=LlmPriority)
    reasoning_effort: str = "none"
    """Sent as `reasoning_effort` ("none": no thinking before a spoken reply; "" omits it)."""

    @model_validator(mode="after")
    def _reserved_below_max(self) -> Self:
        if self.reserved_voice_slots >= self.max_concurrency:
            raise ValueError(
                f"llm.reserved_voice_slots ({self.reserved_voice_slots}) must be less than "
                f"llm.max_concurrency ({self.max_concurrency})"
            )
        return self


class SttConfig(Section):
    """The speech server's speech-to-text (`POST {base_url}/v1/audio/transcriptions`)."""

    base_url: str = "http://127.0.0.1:8772"
    model: str = "parakeet-tdt-0.6b-v3"


class TtsConfig(Section):
    """The speech server's text-to-speech (`POST {base_url}/v1/audio/speech`, streamed PCM).
    The voice is the assistant's (`config/assistants/<id>.toml` `voice.speaker`)."""

    base_url: str = "http://127.0.0.1:8772"
    backend: str = "qwen3-ggml"


class GpuConfig(Section):
    """`[gpu]`: where the stack's model servers run (scripts/llm_server.sh, the speech server
    launchers, the gate). Read by the launchers, never by the brain."""

    layout: Literal["two", "one"] = "two"
    """`two`: the LLM server on GPU0 and the speech server on GPU1 (the brain PC). `one`: both
    on `gpu_index`; the speech server starts first and vLLM is sized to what is left."""
    gpu_index: int = Field(default=0, ge=0)
    """Layout `one`: the card both servers share (nvidia-smi numbering)."""
    margin_mib: int = Field(default=768, ge=0)
    """Layout `one`: GPU memory that stays free at the peak of a busy turn (both servers at
    work); the LLM server sizes its KV cache to leave it and refuses to start if the rest
    would not hold the model."""
    max_model_len: int = Field(default=65536, ge=4096)
    """Layout `one`: vLLM's context length (bf16 KV cache; 64k needs about 2.3 GB of it, which
    leaves room for a small speech server only: 0.6B Q8_0 TTS on a 24 GB card)."""
    max_num_seqs: int = Field(default=8, ge=1)
    """Layout `one`: vLLM's concurrent sequences (they share the one KV cache)."""

    @property
    def llm_gpu(self) -> int:
        return self.gpu_index if self.layout == "one" else 0

    @property
    def speech_gpu(self) -> int:
        return self.gpu_index if self.layout == "one" else 1


class EdgeAudioConfig(Section):
    input: str = "default"
    output: str = "default"
    aec: Literal["hw", "sw", "none"] = "none"


class EdgeListenConfig(Section):
    mode: Literal["wake_word", "open_mic", "push_to_talk"] = "wake_word"
    """`wake_word` (hands-free after a wake word), `open_mic` (any speech starts a turn) or
    `push_to_talk` (debugging: only `/ptt down` ... `/ptt up`)."""


class EdgeVadConfig(Section):
    """The speech detector (Silero VAD) that opens and ends windows in the listening modes."""

    threshold: float = Field(default=0.5, gt=0, lt=1)
    start_ms: int = Field(default=160, ge=20)
    """Speech this long in a row opens a window (open mic, follow-up)."""
    end_ms: int = Field(default=700, ge=100)
    """A window ends after this much no speech once speech was heard."""
    pre_roll_ms: int = Field(default=600, ge=0, le=3000)
    barge_in_stop_delay_ms: int = Field(default=350, ge=0, le=1000)
    """A barge-in lets the robot finish its word: it goes on speaking this long (0.25-0.5 s
    sounds natural), then stops with a short fade (0: it stops at once)."""


class EdgeWakeConfig(Section):
    engine: Literal["openwakeword", "phrase"] = "openwakeword"
    models: list[str] = Field(default_factory=lambda: ["hey_jarvis"], min_length=1)
    """openWakeWord model names (or `.onnx` paths), or one phrase for `phrase`. The brain's
    `welcome` sets each word's threshold."""
    threshold: float = Field(default=0.4, gt=0, le=1)
    """Until the brain's `welcome` arrives."""
    pre_roll_s: float = Field(default=0.1, ge=0, le=3)
    """Audio from before the wake detection that the window starts with. The detection comes
    about 0.1 s after the wake word ends: more would send the wake word to the brain."""
    no_speech_s: float = Field(default=8, gt=0, le=15)
    """A wake window ends if no speech follows the wake word this long (a bare wake: the
    brain answers "Yes?")."""
    phrase_model_dir: str | None = None
    max_window_s: float = Field(default=120, gt=0, le=120)
    """Hard cap of every mic window, whoever opens it (the edge clamps longer requests): at
    most 2 minutes."""


class EdgeConfig(Section):
    device_id: str = "lite"
    room: str = "living"
    brain_url: str = "mdns"
    """`mdns` to discover the brain, or a static `wss://host:port/edge/v1` URL."""
    body: Literal["reachy", "console"] = "console"
    """Body driver: the Reachy Mini, or the console (mic and speaker of this machine)."""
    audio: EdgeAudioConfig = Field(default_factory=EdgeAudioConfig)
    listen: EdgeListenConfig = Field(default_factory=EdgeListenConfig)
    vad: EdgeVadConfig = Field(default_factory=EdgeVadConfig)
    wake: EdgeWakeConfig = Field(default_factory=EdgeWakeConfig)


class ObservabilityConfig(Section):
    otlp: str = ""
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    log_format: Literal["json", "console"] = "json"
    capture_content: bool = False


class EdgeHostConfig(Section):
    ssh: str = ""
    """SSH alias (from ~/.ssh/config) of the machine the robot is plugged into, when it is not
    this one. Empty: the robot is local. The env var ASSISTANT_EDGE_HOST overrides it."""


class HarnessConfig(Section):
    """`[test]`: test-harness settings (read by assistant_testing, never by the runtime)."""

    edge_host: EdgeHostConfig = Field(default_factory=EdgeHostConfig)


class AssistantConfig(BaseSettings):
    """The whole runtime config. Build it with `load_config`."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_DELIMITER,
        case_sensitive=False,
        extra="forbid",
        frozen=True,
    )

    net: NetConfig = Field(default_factory=NetConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    brain: BrainConfig = Field(default_factory=BrainConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    edge: EdgeConfig = Field(default_factory=EdgeConfig)
    body: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Per-body driver settings, e.g. `[body.reachy]`."""
    skills: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Per-skill settings, e.g. `[skills.weather]`; each skill validates its own table."""
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    test: HarnessConfig = Field(default_factory=HarnessConfig)

    @classmethod
    def settings_customise_sources(
        cls, settings_cls: type[BaseSettings], init_settings: Any, *_: Any, **__: Any
    ) -> tuple[Any, ...]:
        # load_config merges files, env and CLI itself; the model only validates the result.
        return (init_settings,)


# ---------------------------------------------------------------- assistants


class AssistantWakeWord(Section):
    model: str
    threshold: float = Field(ge=0.0, le=1.0)
    word: str | None = None
    """Spoken form; defaults to the model name with underscores as spaces."""

    @property
    def spoken(self) -> str:
        return self.word or self.model.replace("_", " ")


class VoiceConfig(Section):
    speaker: str


class SkillPolicy(Section):
    allow: list[str] = Field(default_factory=lambda: ["*"])
    deny: list[str] = Field(default_factory=list)


class MemoryPolicy(Section):
    history_turns: int = Field(default=20, ge=0)
    summarize_after: int = Field(default=40, ge=0)


class AssistantDef(Section):
    """One assistant, from `config/assistants/<id>.toml`."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    wake_words: list[AssistantWakeWord] = Field(min_length=1)
    voice: VoiceConfig
    persona: str
    personas: list[str] = Field(default_factory=list)
    skills: SkillPolicy = Field(default_factory=SkillPolicy)
    memory: MemoryPolicy = Field(default_factory=MemoryPolicy)


# ---------------------------------------------------------------- loading


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`; tables merge, other values replace."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def env_layer(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The `ASSISTANT__…` variables as a nested dict (parsed by pydantic-settings)."""
    source = EnvSettingsSource(
        AssistantConfig,
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_DELIMITER,
        case_sensitive=False,
    )
    if environ is not None:
        source.env_vars = {k.lower(): v for k, v in environ.items()}
    return source()


def parse_cli_overrides(items: Iterable[str]) -> dict[str, Any]:
    """Turn `["llm.model=qwen", "llm.max_concurrency=4"]` into a nested dict.

    Values are read as TOML values when possible (numbers, booleans, arrays, quoted strings)
    and as plain strings otherwise.
    """
    result: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"override must look like section.key=value, got {item!r}")
        try:
            value: Any = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw
        node: dict[str, Any] = {}
        leaf = node
        parts = key.strip().split(".")
        for part in parts[:-1]:
            leaf[part] = {}
            leaf = leaf[part]
        leaf[parts[-1]] = value
        result = deep_merge(result, node)
    return result


def load_config(
    config_dir: Path | str = "config",
    *,
    profile: str | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> AssistantConfig:
    """Load `assistant.toml`, overlay the profile, the environment and CLI overrides.

    `environ` defaults to `os.environ`; pass a mapping to isolate tests.
    """
    config_dir = Path(config_dir)
    layers: list[Mapping[str, Any]] = []
    base = config_dir / "assistant.toml"
    if base.exists():
        layers.append(read_toml(base))
    if profile:
        layers.append(read_toml(config_dir / "profiles" / f"{profile}.toml"))
    layers.append(env_layer(os.environ if environ is None else environ))
    if cli_overrides:
        layers.append(cli_overrides)
    merged: dict[str, Any] = {}
    for layer in layers:
        merged = deep_merge(merged, layer)
    return AssistantConfig(**merged)


def load_assistant(config_dir: Path | str, assistant_id: str) -> AssistantDef:
    path = Path(config_dir) / "assistants" / f"{assistant_id}.toml"
    data = read_toml(path)
    assistant = AssistantDef.model_validate(data)
    if assistant.id != assistant_id:
        raise ConfigError(f"{path}: id {assistant.id!r} does not match the file name")
    return assistant
