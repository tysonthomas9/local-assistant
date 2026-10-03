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


@pytest.mark.skipif(not SCRIPT.exists(), reason="run_poc.sh is not in the gadget image")
def test_bridge_marked_started_before_it_starts():
    """Cleanup stops a half-started bridge only if bridge_started=1 is set before the start command."""
    lines = SCRIPT.read_text().splitlines()
    fake_start = next(i for i, l in enumerate(lines) if "fake_bridge.py" in l and "exec" in l)
    real_start = next(i for i, l in enumerate(lines) if '"$ROOT/mac_gadget.sh" start' in l)
    fake_branch = next(i for i, l in enumerate(lines) if 'if [ "$fake" = 1 ]; then' in l and i < fake_start)
    real_branch = next(i for i in range(fake_start, real_start) if lines[i].strip() == "else")
    for branch, start in ((fake_branch, fake_start), (real_branch, real_start)):
        marks = [i for i in range(branch, start) if lines[i].strip().startswith("bridge_started=1")]
        assert marks, f"no bridge_started=1 between lines {branch + 1} and {start + 1}"


@pytest.mark.skipif(not SCRIPT.exists(), reason="run_poc.sh is not in the gadget image")
def test_gadget_log_is_saved_redacted_before_the_container_goes():
    lines = SCRIPT.read_text().splitlines()
    save = next(i for i, l in enumerate(lines) if '"$ROOT/mac_gadget.sh" logs' in l)
    stop = next(i for i, l in enumerate(lines) if '"$ROOT/mac_gadget.sh" stop' in l)
    assert save < stop
    assert "| redact" in lines[save] and '"$LOGDIR/$RUN.gadget.log"' in lines[save]


@pytest.mark.skipif(not SCRIPT.exists(), reason="run_poc.sh is not in the gadget image")
def test_redact_backstops_text_and_content_anywhere():
    import subprocess
    redact = next(l for l in SCRIPT.read_text().splitlines() if l.startswith("redact()"))
    sample = (
        "2026-10-03 14:05:31,330 INFO gadget.link: client.invoke command=reachy.dance id=c-1\n"
        "2026-10-03 14:05:31,352 INFO gadget.chat: got a 34-character reply in 1 message(s)\n"
        "2026-10-03 14:05:32,000 INFO some.lib: message text=dance for me please\n"
        "2026-10-03 14:05:32,001 INFO some.lib: payload content=Watch this! role=x\n"
    )
    out = subprocess.run(["bash", "-c", redact + "\nredact"], input=sample, capture_output=True,
                         text=True, check=True).stdout
    assert "dance for me" not in out and "Watch this" not in out
    assert "text=<redacted>" in out and "content=<redacted>" in out
    assert "client.invoke command=reachy.dance id=c-1" in out and "34-character reply" in out
