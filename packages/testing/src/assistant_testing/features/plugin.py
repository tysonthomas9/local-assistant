"""pytest plugin: every scenario in `**/features/**/*.yaml` becomes one test item.

Registered through the `pytest11` entry point, so it is active wherever assistant-testing is
installed. The feature's `tier` becomes a marker (`core`, `hw` or `models`; a list such as
`[hw, models]` adds each), so `pytest e2e -m hw` runs only the robot features.
`--list-features` prints the features and their steps without running anything.
"""

import asyncio
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import assistant_testing.steps  # noqa: F401  (registers the built-in steps)
from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.loader import Feature, FeatureError, Scenario, load_feature
from assistant_testing.fixtures import reachy_daemon  # noqa: F401  (registers the fixture)

PASS = "✓"
FAIL = "✗"
SKIPPED = "-"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("features", "YAML feature files")
    group.addoption(
        "--list-features",
        action="store_true",
        default=False,
        help="list collected feature files, scenarios and steps, then exit without running",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "feature: a scenario from a YAML feature file")
    for tier in ("core", "hw", "models"):
        config.addinivalue_line("markers", f"{tier}: feature tier {tier}")


def _is_feature_file(path: Path) -> bool:
    return path.suffix in {".yaml", ".yml"} and "features" in path.parts


def pytest_collect_file(file_path: Path, parent: pytest.Collector) -> pytest.Collector | None:
    if _is_feature_file(file_path):
        return FeatureFile.from_parent(parent, path=file_path)
    return None


class FeatureFile(pytest.File):
    feature: Feature | None = None

    def collect(self) -> list[pytest.Item]:
        try:
            self.feature = load_feature(self.path)
        except FeatureError as exc:
            raise self.CollectError(str(exc)) from None
        items: list[pytest.Item] = []
        for scenario in self.feature.scenarios:
            item = ScenarioItem.from_parent(
                self, name=scenario.name, feature=self.feature, scenario=scenario
            )
            item.add_marker("feature")
            for tier in self.feature.tiers:
                item.add_marker(tier)
            items.append(item)
        return items


@dataclass
class StepResult:
    mark: str
    text: str
    line: int
    seconds: float = 0.0


class StepFailed(Exception):
    def __init__(self, step_text: str, line: int, cause: BaseException) -> None:
        super().__init__(f"{step_text} (line {line}): {cause}")
        self.cause = cause


class ScenarioItem(pytest.Item):
    def __init__(self, *, feature: Feature, scenario: Scenario, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.feature = feature
        self.scenario = scenario
        self.results: list[StepResult] = []

    def reportinfo(self) -> tuple[Path, int, str]:
        return self.path, self.scenario.line - 1, f"{self.feature.name}: {self.scenario.name}"

    def runtest(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        self.results = []
        ctx = ScenarioContext(repo_root=self.config.rootpath, feature_path=self.path)
        failure: StepFailed | None = None
        try:
            for index, bound in enumerate(self.scenario.steps):
                started = time.perf_counter()
                try:
                    await bound.definition.fn(ctx, **bound.kwargs)
                except Exception as exc:
                    elapsed = time.perf_counter() - started
                    self.results.append(StepResult(FAIL, bound.text, bound.line, elapsed))
                    self.results.extend(
                        StepResult(SKIPPED, later.text, later.line)
                        for later in self.scenario.steps[index + 1 :]
                    )
                    failure = StepFailed(bound.text, bound.line, exc)
                    failure.__traceback__ = exc.__traceback__
                    break
                self.results.append(
                    StepResult(PASS, bound.text, bound.line, time.perf_counter() - started)
                )
        finally:
            await ctx.aclose()
        if failure is not None:
            raise failure

    def step_lines(self) -> list[str]:
        lines: list[str] = []
        for r in self.results:
            timing = f"  ({r.seconds * 1000:.0f} ms)" if r.mark != SKIPPED else ""
            lines.append(f"    {r.mark} {r.text}{timing}")
        return lines

    def repr_failure(
        self,
        excinfo: pytest.ExceptionInfo[BaseException],
        style: Any = None,
    ) -> str:
        header = f"{self.feature.name}: {self.scenario.name}  ({self.path.name})"
        lines = [header, *self.step_lines()]
        exc = excinfo.value
        if isinstance(exc, StepFailed):
            cause = exc.cause
            lines.append("")
            lines.append(f"{FAIL} {exc}")
            if not isinstance(cause, AssertionError | TimeoutError):
                lines.append("".join(traceback.format_exception(cause)).rstrip())
        else:
            lines.append(str(excinfo.getrepr(style="short")))
        return "\n".join(lines)


def pytest_runtest_logfinish(nodeid: str, location: tuple[str, int | None, str]) -> None:
    del location
    item = _ITEMS.get(nodeid)
    if item is None or not item.results:
        return
    reporter = item.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    reporter.ensure_newline()
    for line in item.step_lines():
        reporter.write_line(line)


_ITEMS: dict[str, ScenarioItem] = {}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if isinstance(item, ScenarioItem):
            _ITEMS[item.nodeid] = item


def pytest_collection_finish(session: pytest.Session) -> None:
    if not session.config.getoption("list_features"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    write = reporter.write_line if reporter is not None else print
    features: dict[Path, Feature] = {}
    for item in session.items:
        if isinstance(item, ScenarioItem):
            features.setdefault(item.path, item.feature)
    write("")
    scenarios = sum(len(f.scenarios) for f in features.values())
    write(f"{len(features)} feature(s), {scenarios} scenario(s)")
    for path, feature in features.items():
        rel = path.relative_to(session.config.rootpath)
        write(f"[{feature.tier}] {feature.name}  ({rel})")
        write(f"    {feature.description}")
        for scenario in feature.scenarios:
            write(f"  - {scenario.name}")
            for bound in scenario.steps:
                write(f"      {bound.text}")


def pytest_runtestloop(session: pytest.Session) -> bool | None:
    if session.config.getoption("list_features"):
        return True
    return None
