"""run_poc.sh defaults that the user asked for."""

import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_poc.sh"


@pytest.mark.skipif(not SCRIPT.exists(), reason="run_poc.sh is not in the gadget image")
def test_default_volume_is_100():
    text = SCRIPT.read_text()
    assert re.search(r"(^|[; ])volume=100;", text, re.M)
    assert "default 100;" in text


@pytest.mark.skipif(not SCRIPT.exists(), reason="run_poc.sh is not in the gadget image")
@pytest.mark.parametrize("args", [["--duration", "-5"], ["--duration", "1.5"], ["--lock-timeout", "x"],
                                  ["--mic-log", "1;id"], ["--mic-log", ""]])
def test_bad_numbers_refused_before_anything_starts(args):
    import subprocess
    done = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=10)
    assert done.returncode == 2
    assert "must be a whole number" in done.stderr
