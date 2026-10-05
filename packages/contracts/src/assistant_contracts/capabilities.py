"""Edge capabilities, sent in `hello.body.capabilities` (proposal section 5.1)."""

from typing import Literal

from pydantic import Field

from assistant_contracts.common import Aec, ContractModel


class AudioInCaps(ContractModel):
    rate: int = 16000
    aec: Aec = "none"


class AudioOutCaps(ContractModel):
    rates: list[int] = Field(default_factory=lambda: [16000, 24000])


class WakeCaps(ContractModel):
    engines: list[str] = Field(default_factory=list)


class MotionCaps(ContractModel):
    expressions: list[str] = Field(default_factory=list)
    look_at: list[Literal["user", "doa", "world", "image"]] = Field(default_factory=list)
    attention: bool = False
    sequence: bool = False
    """Minor feature `motion.sequence`."""


class CameraCaps(ContractModel):
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class Capabilities(ContractModel):
    """What an edge (and its body) can do. Optional parts are None when absent."""

    audio_in: AudioInCaps = Field(default_factory=AudioInCaps)
    audio_out: AudioOutCaps = Field(default_factory=AudioOutCaps)
    wake: WakeCaps | None = None
    motion: MotionCaps | None = None
    camera: CameraCaps | None = None
    doa: bool = False
    opus: bool = False
    """The edge can send and receive 0x05 Opus frames (reserved in v1; off by default)."""
    speak_text: bool = False
    """The edge can use reply text sent in `speak.begin.text`."""


BodyCapabilities = Capabilities
"""What `Body.start()` returns; the edge forwards it as `hello.body.capabilities`."""
