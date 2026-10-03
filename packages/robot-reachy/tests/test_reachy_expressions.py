"""`[body.reachy] expressions`: a TOML `[moves]` table over the built-in emotion moves."""

from pathlib import Path

import pytest

from assistant_robot_reachy.body import expression_moves

pytestmark = pytest.mark.unit


def test_repo_expressions_file_is_valid() -> None:
    assert expression_moves(str(Path(__file__).parents[3] / "config/bodies/reachy.toml")) == {}


def test_expressions_file(tmp_path: Path) -> None:
    file = tmp_path / "reachy.toml"
    file.write_text('[moves]\nhappy = "enthusiastic1"\n')
    assert expression_moves(str(file)) == {"happy": "enthusiastic1"}
    assert expression_moves(None) == {}
    with pytest.raises(ValueError, match="no file"):
        expression_moves(str(tmp_path / "missing.toml"))
