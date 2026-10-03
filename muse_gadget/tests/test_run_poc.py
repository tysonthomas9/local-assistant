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
