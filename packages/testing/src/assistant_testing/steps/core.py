"""Workspace-level steps: packages import, config loads."""

import json
import sys
from pathlib import Path
from typing import Any

from assistant_core.config import load_assistant, load_config
from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.registry import step

PACKAGES = (
    "assistant_contracts",
    "assistant_core",
    "assistant_link",
    "assistant_skill_sdk",
    "assistant_brain",
    "assistant_skills_builtin",
    "assistant_skills_web",
    "assistant_edge",
    "assistant_robot_reachy",
    "assistant_cli",
    "assistant_testing",
)

_IMPORT_SCRIPT = """
import importlib, json, sys
out = {}
for name in sys.argv[1:]:
    out[name] = importlib.import_module(name).__file__
print(json.dumps(out))
"""


@step("import_packages")
async def import_packages(ctx: ScenarioContext, packages: list[str] | None = None) -> None:
    """Import packages in a fresh Python process (default: all 11) from this checkout."""
    names = packages or list(PACKAGES)
    run = await ctx.processes.run(
        "import", [sys.executable, "-c", _IMPORT_SCRIPT, *names], timeout_s=60
    )
    assert run.returncode == 0, f"import failed:\n{run.output}"
    files: dict[str, str] = json.loads(run.output.strip().splitlines()[-1])
    assert sorted(files) == sorted(names)
    root = ctx.repo_root.resolve()
    for name, file in files.items():
        assert Path(file).resolve().is_relative_to(root), (
            f"{name} was imported from {file}, not from this checkout"
        )


@step("load_config")
async def load_config_step(
    ctx: ScenarioContext, profile: str | None = None, env: dict[str, str] | None = None
) -> None:
    """Load config/ with an optional profile. Only `env` is used as the environment."""
    ctx.state["config"] = load_config(ctx.repo_root / "config", profile=profile, environ=env or {})


@step("expect_config")
async def expect_config(ctx: ScenarioContext, path: str, equals: Any) -> None:
    """Compare a dotted config path (e.g. `llm.impl`) of the last loaded config."""
    assert "config" in ctx.state, "no config loaded yet; use load_config first"
    value: Any = ctx.state["config"]
    for part in path.split("."):
        value = value[part] if isinstance(value, dict) else getattr(value, part)
    if isinstance(value, Path):
        value = str(value)
    assert value == equals, f"{path} is {value!r}, expected {equals!r}"


@step("load_assistant")
async def load_assistant_step(ctx: ScenarioContext, id: str) -> None:
    """Load config/assistants/<id>.toml and check it validates."""
    assistant = load_assistant(ctx.repo_root / "config", id)
    ctx.state.setdefault("assistants", {})[id] = assistant
