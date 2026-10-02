"""The robot-move measurement: sampling density, the move interval, head turn (pure logic)."""

import math
from typing import Any

import pytest

from assistant_testing.steps.edge import check_sampling, head_turn_deg, move_samples


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


def test_the_move_interval_picks_its_samples_and_the_start_pose() -> None:
    samples = [_state(i * 0.05, pitch_deg=i) for i in range(20)]
    ref, during = move_samples(samples, t_start=0.22, t_end=0.61)
    assert ref is samples[4]
    assert during == samples[5:13]


def test_a_move_interval_without_samples_fails() -> None:
    samples = [_state(i * 0.05) for i in range(5)]
    with pytest.raises(AssertionError, match="while the move played"):
        move_samples(samples, t_start=0.21, t_end=0.22)
    with pytest.raises(AssertionError, match="before the move started"):
        move_samples(samples, t_start=-1.0, t_end=0.1)


def _pose(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> dict[str, float]:
    return {"roll": math.radians(roll), "pitch": math.radians(pitch), "yaw": math.radians(yaw)}


def test_head_turn_is_the_rotation_angle_whatever_the_axis() -> None:
    assert head_turn_deg(_pose(), _pose(pitch=12)) == pytest.approx(12)
    assert head_turn_deg(_pose(), _pose(roll=-15)) == pytest.approx(15)
    assert head_turn_deg(_pose(pitch=3, roll=2), _pose(pitch=3, roll=2)) == pytest.approx(
        0, abs=1e-6
    )
    assert head_turn_deg(_pose(), _pose(pitch=6, roll=8)) == pytest.approx(10, abs=0.2)
