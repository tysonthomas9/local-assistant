"""Small types shared by EdgeLink messages, the Body Protocols and bus events."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Channel = Literal["speech", "alert", "media"]
"""Audio output channel; AudioFocus priority is speech > alert > media."""

AttentionState = Literal["idle", "listening", "thinking", "speaking", "muted", "sleeping"]

Aec = Literal["hw", "sw", "none"]

StreamId = Annotated[int, Field(ge=0, le=255)]
"""Output stream id. It is the `stream` byte of binary frames, so it fits in a u8."""

Slot = Annotated[int, Field(ge=0, le=255)]
"""Snapshot slot. It is the `stream` byte of 0x03 frames, so it fits in a u8."""

Score = Annotated[float, Field(ge=0.0, le=1.0)]


class ContractModel(BaseModel):
    """Base for contract models: immutable; unknown fields ignored (minor-version tolerance)."""

    model_config = ConfigDict(frozen=True, extra="ignore")


class LookAtUser(ContractModel):
    kind: Literal["user"] = "user"
    follow: bool = Field(
        default=True,
        description="Keep following the user (face tracking on), or stop following (off).",
    )
    """Added in v1 as an optional field with the old meaning as its default: no version bump."""


class LookAtDoa(ContractModel):
    kind: Literal["doa"] = "doa"
    doa: float = Field(description="Direction of arrival in degrees, 0 = straight ahead.")


class LookAtWorld(ContractModel):
    kind: Literal["world"] = "world"
    x: float
    y: float
    z: float


class LookAtImage(ContractModel):
    kind: Literal["image"] = "image"
    u: float = Field(ge=0.0, le=1.0, description="Horizontal position, 0 = left edge.")
    v: float = Field(ge=0.0, le=1.0, description="Vertical position, 0 = top edge.")


LookTarget = Annotated[
    LookAtUser | LookAtDoa | LookAtWorld | LookAtImage, Field(discriminator="kind")
]
"""Where to look: the user, a direction of arrival, a world point or an image point."""
