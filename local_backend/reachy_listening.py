"""Privacy mute for the `listening` tool: stop sending microphone audio, for a while or until resumed.

Muting sets LocalStream._mic_muted through reachy_bridge; the app's record loop then drops every
mic frame before it is sent (console.py record_loop), so nothing reaches the speech server. The
robot can still speak (confirmations, reminders, radio). It un-mutes when the timer runs out, from
the web UI's mic toggle, or by the wake word (unless muted with hard=True).

While muted the robot holds a "not listening" pose: antennas down (values from Pollen's recorded
sad/downcast emotions) and head tilted down 12°. The pose subclasses the app's BreathingMove, so
dances and emotions can still take over; a watcher re-applies it once the robot is idle again.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

import reachy_bridge

logger = logging.getLogger("reachy_listening")

MUTED_ANTENNAS = np.array([-2.2, 2.2])
MUTED_PITCH_DEG = 12.0
CUE_CHECK_S = 3.0

_lock = threading.Lock()
_state: dict[str, Any] = {"until": None, "hard": False, "generation": 0, "by_us": False, "wake": None}


def _muted_pose_class():
    from reachy_mini.utils import create_head_pose
    from reachy_mini_conversation_app.moves import BreathingMove

    class MutedPose(BreathingMove):  # BreathingMove: other moves may pre-empt it
        def __init__(self, start_pose, start_antennas):
            super().__init__(start_pose, start_antennas, interpolation_duration=1.5)
            self.neutral_head_pose = create_head_pose(pitch=MUTED_PITCH_DEG, degrees=True)
            self.neutral_antennas = MUTED_ANTENNAS

        def evaluate(self, t):
            if t < self.interpolation_duration:
                return super().evaluate(t)
            return self.neutral_head_pose, self.neutral_antennas.astype(np.float64), 0.0

    return MutedPose


def _current_pose(mm: Any) -> tuple[Any, Any]:
    head, antennas = None, None
    try:
        last = getattr(mm.state, "last_primary_pose", None)
        if last:
            head, antennas = last[0], last[1]
    except Exception:
        pass
    from reachy_mini.utils import create_head_pose
    return (head if head is not None else create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
            tuple(antennas) if antennas is not None else (-0.17, 0.17))


def _cue_watcher(mm: Any, generation: int) -> None:
    """Keep the muted pose up while muted: re-queue it whenever the robot is back to idle breathing."""
    try:
        MutedPose = _muted_pose_class()
        from reachy_mini_conversation_app.moves import BreathingMove
    except Exception as e:
        logger.warning("No muted pose (%r)", e)
        return
    while True:
        with _lock:
            if _state["generation"] != generation or not reachy_bridge.mic_muted():
                break
        try:
            cur = mm.state.current_move
            if (cur is None or type(cur) is BreathingMove) and not mm.move_queue:
                mm.queue_move(MutedPose(*_current_pose(mm)))
        except Exception as e:
            logger.debug("muted pose check failed: %r", e)
        time.sleep(CUE_CHECK_S)
    try:  # leave the pose only if we're really listening again (a re-mute keeps the new pose)
        if not reachy_bridge.mic_muted() and type(mm.state.current_move).__name__ == "MutedPose":
            mm.clear_move_queue()
    except Exception:
        pass


def _auto_resume(generation: int, seconds: float, mm: Any, wake: threading.Event) -> None:
    wake.wait(seconds)            # set early when this mute is superseded, so the thread doesn't linger
    with _lock:
        if _state["generation"] != generation:
            return
    logger.info("Mute timer ended; listening again")
    resume(mm, reason="timer")


def mute(minutes: float | None, hard: bool, mm: Any = None) -> dict[str, Any]:
    if reachy_bridge.stream() is None:
        return {"error": "The conversation app isn't running."}
    with _lock:
        _supersede()
        gen = _state["generation"]
        _state["hard"] = bool(hard)
        _state["until"] = time.time() + minutes * 60 if minutes else None
        _state["by_us"] = True
        _state["wake"] = wake = threading.Event()
        # flip the mic under the same lock: _sync_with_mic (every mic frame with --wake) must never see our
        # state without the mic muted, or it would take it for a UI un-mute and drop the timer/hard flag
        reachy_bridge.set_mic_muted(True)
    if minutes:
        threading.Thread(target=_auto_resume, args=(gen, minutes * 60, mm, wake), daemon=True, name="mute-timer").start()
    if mm is not None:
        threading.Thread(target=_cue_watcher, args=(mm, gen), daemon=True, name="muted-pose").start()
    return status()


def _supersede() -> None:
    """New generation; wake the previous mute's timer so it exits now. Call with _lock held."""
    _state["generation"] += 1
    if _state["wake"] is not None:
        _state["wake"].set()
        _state["wake"] = None


def _sync_with_mic() -> None:
    """The web UI's mic toggle sets _mic_muted directly; reconcile our state with it."""
    with _lock:
        muted = reachy_bridge.mic_muted()
        if not muted and (_state["by_us"] or _state["hard"] or _state["until"]):
            _supersede()   # un-muted from the UI: forget our mute (timer, hard flag)
            _state.update(until=None, hard=False, by_us=False)
        elif muted and not _state["by_us"]:
            _state.update(until=None, hard=False)   # muted from the UI: a plain (soft) mute


def resume(mm: Any = None, reason: str = "asked") -> dict[str, Any]:
    with _lock:
        _supersede()
        _state["until"], _state["hard"], _state["by_us"] = None, False, False
        reachy_bridge.set_mic_muted(False)
    return {**status(), "resumed_by": reason}


def status() -> dict[str, Any]:
    _sync_with_mic()
    muted = reachy_bridge.mic_muted()
    out: dict[str, Any] = {"listening": muted is False, "muted": bool(muted)}
    if muted and _state["until"]:
        left = max(0, _state["until"] - time.time())
        out["resumes_at"] = time.strftime("%I:%M %p", time.localtime(_state["until"])).lstrip("0")
        out["minutes_left"] = round(left / 60)
    if muted:
        out["hard"] = _state["hard"]
    return out


def is_hard_muted() -> bool:
    _sync_with_mic()
    return bool(reachy_bridge.mic_muted()) and bool(_state["hard"])
