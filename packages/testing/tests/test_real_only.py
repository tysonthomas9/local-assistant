"""The real-only checker finds test doubles in e2e files and names file and line."""

from pathlib import Path

import pytest

from assistant_testing import real_only
from assistant_testing.real_only import check, main

pytestmark = pytest.mark.unit

DOUBLE = "Fa" + "keLLM"  # built from pieces so this file itself stays out of the way


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("from unittest.mock import AsyncMock\n", "imports unittest.mock"),
        ("from unittest import mock\n", "imports unittest.mock"),
        ("import pytest_mock\n", "imports pytest_mock"),
        ("def test(monkeypatch): ...\n", "uses monkeypatch"),
        ("def test_x(mocker): ...\n", "uses the pytest-mock mocker fixture"),
        (f"llm = {DOUBLE}()\n", "test-double identifier"),
        ("class " + "Stu" + "bBody: ...\n", "test-double identifier"),
        ("- start_edge: {body: " + "Dum" + "myBody}\n", "test-double identifier"),
    ],
)
def test_violations_are_reported_with_line(tmp_path: Path, text: str, rule: str) -> None:
    target = tmp_path / "e2e" / "features" / "x.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("ok: line\n" + text)
    (violation,) = check(tmp_path)
    assert (violation.path, violation.line, violation.rule) == (target, 2, rule)
    assert main([str(tmp_path)]) == 1


def test_plain_words_pass(tmp_path: Path) -> None:
    steps = tmp_path / "packages/testing/src/assistant_testing/steps"
    steps.mkdir(parents=True)
    (steps / "s.py").write_text("# no fakes, mocks or monkeypatching here; Faker is fine\n")
    assert check(tmp_path) == []
    assert main([str(tmp_path)]) == 0


def test_mocker_fixture_under_e2e_fails(tmp_path: Path) -> None:
    target = tmp_path / "e2e" / "test_x.py"
    target.parent.mkdir(parents=True)
    target.write_text("def test_x(mocker):\n    pass\n")
    (violation,) = check(tmp_path)
    assert (violation.path, violation.line) == (target, 1)
    assert main([str(tmp_path)]) == 1


def test_whole_testing_package_is_scanned(tmp_path: Path) -> None:
    pkg = tmp_path / "packages/testing/src/assistant_testing"
    pkg.mkdir(parents=True)
    target = pkg / "processes.py"
    target.write_text("import asyncio\nfrom unittest.mock import patch\n")
    (violation,) = check(tmp_path)
    assert (violation.path, violation.line, violation.rule) == (target, 2, "imports unittest.mock")
    assert main([str(tmp_path)]) == 1


def test_only_the_checkers_own_rule_strings_are_allowlisted(tmp_path: Path) -> None:
    pkg = tmp_path / "packages/testing/src/assistant_testing"
    pkg.mkdir(parents=True)
    source = Path(real_only.__file__).read_text()
    (pkg / "real_only.py").write_text(source)
    assert check(tmp_path) == []
    (pkg / "real_only.py").write_text(source + "\nimport unittest.mock\n")
    (violation,) = check(tmp_path)
    assert violation.rule == "imports unittest.mock"


@pytest.mark.parametrize(
    "line",
    [
        'LABEL = "imports unittest.mock"  # import unittest.mock\n',
        "# the rule label imports pytest_mock, mentioned in a comment\n",
        'x = "uses monkeypatch" + "monkeypatch"\n',
    ],
)
def test_allowlist_matches_exact_rule_literals_only(tmp_path: Path, line: str) -> None:
    pkg = tmp_path / "packages/testing/src/assistant_testing"
    pkg.mkdir(parents=True)
    (pkg / "real_only.py").write_text(Path(real_only.__file__).read_text() + line)
    assert len(check(tmp_path)) == 1
