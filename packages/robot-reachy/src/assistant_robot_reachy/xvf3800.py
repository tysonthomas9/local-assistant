"""Read the echo-cancellation state of the Reachy Mini's XVF3800 audio board (over USB control).

The robot's microphone path is the XVF3800's processed output. Its acoustic echo canceller
(AEC) uses what the board itself plays as the far-end reference, so it applies to everything
played through the robot's speaker. `AEC_AECCONVERGED` becomes 1 once the canceller has
adapted to the room (it needs some far-end audio first).

Seen on firmware 2.1.2: the board's DSP-side control servicer can get stuck answering "retry"
to every request (AEC, post-processing and audio-manager parameters all unreadable, while
VERSION and DOA still read). A reboot of the audio board alone clears it, with no motion:
`python -m reachy_mini.media.audio_control_utils REBOOT --values 1` (no daemon running).
"""

from typing import Any

AEC_PARAMETERS = (
    "AEC_NUM_FARENDS",
    "AEC_ASROUTONOFF",
    "AEC_AECCONVERGED",
    "AEC_AECPATHCHANGE",
    "AEC_RT60",
    "PP_AGCONOFF",
)


def _plain(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    try:
        return [_plain(v) for v in value]
    except TypeError:
        return str(value)


def aec_status() -> dict[str, Any]:
    """The board's AEC parameters, or `{"board": "not found"}`."""
    from reachy_mini.media.audio_control_utils import init_respeaker_usb

    board = init_respeaker_usb()
    if board is None:
        return {"board": "not found"}
    try:
        status: dict[str, Any] = {"board": "xvf3800"}
        for name in AEC_PARAMETERS:
            try:
                status[name] = _plain(board.read_values(name))
            except Exception as exc:
                status[name] = f"unreadable: {type(exc).__name__}"
        return status
    finally:
        board.close()


STARTUP_TUNING: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("PP_AGCMAXGAIN", (10.0,)),
    ("PP_MIN_NS", (0.8,)),
    ("PP_MIN_NN", (0.8,)),
    ("PP_GAMMA_E", (0.5,)),
    ("PP_GAMMA_ETAIL", (0.5,)),
    ("PP_NLATTENONOFF", (0,)),
    ("PP_MGSCALE", (4.0, 1.0, 1.0)),
)
"""Pollen's conversation-app tuning of the board's post-processing (audio/startup_config.py):
the AGC's maximum gain down to 10, gentler stationary and non-stationary noise suppression,
softer echo suppression (gamma), the non-linear attenuation off. Written at body start; not
persistent across a power cycle of the board."""
WRITE_SETTLE_S = 0.1


def _same(read: Any, expected: tuple[float, ...]) -> bool:
    if not isinstance(read, list) or len(read) != len(expected):
        return False
    return all(abs(float(a) - float(b)) < 1e-3 for a, b in zip(read, expected, strict=True))


def apply_startup_tuning() -> dict[str, Any]:
    """Write `STARTUP_TUNING` (the SDK's `apply_audio_config`, verified), then read every
    parameter back: `{"applied": bool, "readback": {name: values}}`."""
    from reachy_mini.media.audio_control_utils import init_respeaker_usb

    board = init_respeaker_usb()
    if board is None:
        return {"applied": False, "readback": {}, "board": "not found"}
    try:
        try:
            applied = bool(
                board.apply_audio_config(
                    STARTUP_TUNING, verify=True, write_settle_seconds=WRITE_SETTLE_S
                )
            )
        except Exception as exc:
            return {"applied": False, "readback": {}, "error": f"{type(exc).__name__}: {exc}"}
        readback: dict[str, Any] = {}
        for name, _ in STARTUP_TUNING:
            try:
                readback[name] = _plain(board.read_values(name))
            except Exception as exc:
                readback[name] = f"unreadable: {type(exc).__name__}"
        matches = all(_same(readback[name], values) for name, values in STARTUP_TUNING)
        return {"applied": applied and matches, "readback": readback}
    finally:
        board.close()
