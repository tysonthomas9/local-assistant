"""The real-only check: e2e code and feature files may not use test doubles.

E2E uses only real devices and the real stack. This scans `e2e/` and the feature-runner and
step modules of this package, and reports every line that imports `unittest.mock` or
`pytest_mock`, uses `monkeypatch`, or defines/references an identifier like `FakeX`, `MockX`,
`StubX` or `DummyX`.

    python -m assistant_testing.real_only [repo_root]     # exit 1 and file:line on violations
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

SCANNED = (
    "e2e",
    "packages/testing/src/assistant_testing/features",
    "packages/testing/src/assistant_testing/steps",
)
SUFFIXES = {".py", ".yaml", ".yml", ".md", ".toml", ".json", ".txt"}

RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "imports unittest.mock",
        re.compile(r"\bunittest\.mock\b|\bfrom\s+unittest\s+import\s+mock\b"),
    ),
    ("imports pytest_mock", re.compile(r"\bpytest_mock\b")),
    ("uses monkeypatch", re.compile(r"\bmonkeypatch\b")),
    ("test-double identifier", re.compile(r"\b(?:Fake|Mock|Stub|Dummy)[A-Z]\w*")),
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    rule: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.text.strip()}"


def scanned_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in SCANNED:
        base = repo_root / rel
        if base.is_dir():
            files.extend(
                p
                for p in sorted(base.rglob("*"))
                if p.is_file() and p.suffix in SUFFIXES and "__pycache__" not in p.parts
            )
    return files


def check_file(path: Path) -> list[Violation]:
    found: list[Violation] = []
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        for rule, pattern in RULES:
            if pattern.search(line):
                found.append(Violation(path, number, rule, line))
    return found


def check(repo_root: Path) -> list[Violation]:
    return [v for path in scanned_files(repo_root) for v in check_file(path)]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0] if args else ".").resolve()
    files = scanned_files(root)
    violations = [v for path in files for v in check_file(path)]
    for violation in violations:
        print(
            f"{violation.path.relative_to(root)}:{violation.line}: {violation.rule}: "
            f"{violation.text.strip()}"
        )
    if violations:
        print(f"real-only check FAILED: {len(violations)} violation(s) in {len(files)} files")
        return 1
    print(f"real-only check passed: {len(files)} files, no test doubles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
