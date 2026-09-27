"""The feature runner itself: collection errors name file and line; teardown always runs."""

import os
import textwrap
from pathlib import Path

import pytest

from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.loader import FeatureError, load_feature
from assistant_testing.features.registry import step

pytestmark = pytest.mark.unit


@step("_test_start_sleeper")
async def _start_sleeper(ctx: ScenarioContext, pidfile: str) -> None:
    proc = await ctx.processes.start("sleeper", ["sleep", "300"])
    Path(pidfile).write_text(str(proc.pid))


@step("_test_fail")
async def _fail(ctx: ScenarioContext, message: str = "boom") -> None:
    raise AssertionError(message)


@step("_test_edge")
async def _edge(ctx: ScenarioContext, name: str, body: str) -> None:
    del ctx, name, body


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "features" / "f.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    return path


@pytest.mark.parametrize(
    ("body", "line", "text"),
    [
        (
            """\
            feature: x
            tier: core
            description: d
            scenarios:
              - name: s
                steps:
                  - load_config
                  - no_such_step: 1
            """,
            8,
            "unknown step 'no_such_step'",
        ),
        (
            """\
            feature: x
            tier: core
            description: d
            scenarios:
              - name: s
                steps:
                  - roundtrip_frame: {kind: mic_pcm, payload_bytes: lots}
            """,
            7,
            "payload_bytes",
        ),
        (
            """\
            feature: x
            tier: space
            description: d
            scenarios: []
            """,
            2,
            "tier",
        ),
        (
            """\
            feature: x
            tier: core
            description: d
            scenarios:
              - name: s
                steps:
                  - roundtrip_frame: {kind: mic_pcm, colour: red}
            """,
            7,
            "colour",
        ),
    ],
)
def test_errors_name_file_and_line(tmp_path: Path, body: str, line: int, text: str) -> None:
    path = _write(tmp_path, body)
    with pytest.raises(FeatureError) as info:
        load_feature(path)
    assert str(info.value).startswith(f"{path}:{line}:")
    assert text in str(info.value)


@pytest.mark.parametrize(
    ("body", "ok"),
    [("reachy", True), ("console", True), ("null", False), ("simulator", False)],
)
def test_only_real_bodies(tmp_path: Path, body: str, ok: bool) -> None:
    path = _write(
        tmp_path,
        f"""\
        feature: x
        tier: core
        description: d
        scenarios:
          - name: s
            steps:
              - _test_edge:
                  name: desk
                  body: {body}
        """,
    )
    if ok:
        assert load_feature(path).scenarios[0].steps[0].kwargs == {"name": "desk", "body": body}
        return
    with pytest.raises(FeatureError, match=r"f\.yaml:9: .*not a real body type"):
        load_feature(path)


def test_failed_step_still_stops_processes(pytester: pytest.Pytester, tmp_path: Path) -> None:
    pidfile = tmp_path / "sleeper.pid"
    pytester.makepyprojecttoml(
        '[tool.pytest.ini_options]\nasyncio_default_fixture_loop_scope = "function"\n'
    )
    feature = pytester.path / "features" / "teardown.yaml"
    feature.parent.mkdir()
    feature.write_text(
        textwrap.dedent(
            f"""\
            feature: teardown
            tier: core
            description: a failing step after a process start
            scenarios:
              - name: fails after starting a process
                steps:
                  - _test_start_sleeper: {{pidfile: "{pidfile}"}}
                  - _test_fail: expected failure
                  - load_config
            """
        )
    )
    # The plugin is active through its pytest11 entry point; the two test steps above are
    # registered because this module is already imported in this process.
    result = pytester.runpytest_inprocess("features", "-p", "no:cacheprovider")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        ["*✓ _test_start_sleeper*", '*✗ _test_fail: "expected failure"*', "*- load_config*"]
    )
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_reachy_daemon_placeholder_fails_with_a_clear_message(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("def test_robot(reachy_daemon):\n    assert reachy_daemon\n")
    result = pytester.runpytest("-p", "no:cacheprovider", "-p", "no:asyncio")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*NotImplementedError: reachy_daemon is a placeholder*S3*"])
