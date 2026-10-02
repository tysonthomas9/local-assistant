"""The robot-move measurement: sampling density and the neutral window (pure logic)."""

import math
from typing import Any

import pytest

from assistant_testing.steps.edge import _move_window, check_sampling


def _state(t: float, pitch_deg: float = 0.0, mode: str = "enabled") -> dict[str, Any]:
    return {
        "t": t,
        "head_pose": {"roll": 0.0, "pitch": math.radians(pitch_deg), "yaw": 0.0},
        "antennas_position": [0.0, 0.0],
        "control_mode": mode,
    }


def test_dense_sampling_passes() -> None:
    samples = [_state(i * 0.05) for i in range(200)]
    check_sampling(samples, 199 * 0.05)


def test_sparse_sampling_is_reported_as_starved() -> None:
    samples = [_state(i * 2.2) for i in range(6)]
    with pytest.raises(AssertionError, match="starved"):
        check_sampling(samples, 13.2)


def test_one_long_gap_is_reported_as_starved() -> None:
    samples = [_state(i * 0.05) for i in range(100)] + [_state(5.5 + i * 0.05) for i in range(100)]
    with pytest.raises(AssertionError, match=r"starved.*gap"):
        check_sampling(samples, samples[-1]["t"])


def test_from_rest_the_window_spans_first_to_last_neutral_sample() -> None:
    rest = _state(0.0, pitch_deg=28.0, mode="disabled")
    pitches = [28, 20, 5, 0.5, 0.2, 8, 15, 8, 0.3, 0.1, 10, 28]
    samples = [_state(i * 0.05, p) for i, p in enumerate(pitches)]
    ref, window = _move_window(rest, samples)
    assert ref is samples[3]
    assert window == samples[3:10]
    assert max(abs(math.degrees(s["head_pose"]["pitch"])) for s in window) == pytest.approx(15)


def test_from_rest_without_reaching_neutral_fails() -> None:
    rest = _state(0.0, pitch_deg=28.0, mode="disabled")
    with pytest.raises(AssertionError, match="neutral"):
        _move_window(rest, [_state(0.05, 28.0), _state(0.1, 27.0)])


def test_an_awake_robot_is_measured_from_its_start() -> None:
    awake = _state(0.0, pitch_deg=5.0)
    samples = [_state(0.05, 5.0), _state(0.1, 15.0)]
    assert _move_window(awake, samples) == (awake, samples)
