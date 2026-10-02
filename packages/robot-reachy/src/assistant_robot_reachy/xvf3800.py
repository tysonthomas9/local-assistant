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
