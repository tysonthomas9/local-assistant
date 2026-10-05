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
    value = _lookup(ctx.state["config"], path)
    assert value == equals, f"{path} is {value!r}, expected {equals!r}"


def _lookup(value: Any, path: str) -> Any:
    """Follow a dotted path through models, dicts and lists (`wake_words.0.spoken`)."""
    for part in path.split("."):
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list | tuple):
            value = value[int(part)]
        else:
            value = getattr(value, part)
    return str(value) if isinstance(value, Path) else value


@step("load_assistant")
async def load_assistant_step(ctx: ScenarioContext, id: str) -> None:
    """Load config/assistants/<id>.toml and check it validates."""
    assistant = load_assistant(ctx.repo_root / "config", id)
    ctx.state.setdefault("assistants", {})[id] = assistant


@step("expect_assistant")
async def expect_assistant(ctx: ScenarioContext, id: str, path: str, equals: Any) -> None:
    """Compare a dotted path (e.g. `wake_words.0.spoken`) of an assistant loaded earlier."""
    assistants = ctx.state.get("assistants", {})
    assert id in assistants, f"assistant {id!r} not loaded yet; use load_assistant first"
    value = _lookup(assistants[id], path)
    assert value == equals, f"{id}.{path} is {value!r}, expected {equals!r}"


@step("real_only_check")
async def real_only_check(ctx: ScenarioContext) -> None:
    """Run the real-only check (python -m assistant_testing.real_only) over this checkout."""
    run = await ctx.processes.run(
        "real-only",
        [sys.executable, "-m", "assistant_testing.real_only", str(ctx.repo_root)],
        timeout_s=60,
    )
    assert run.returncode == 0, run.output
    assert "real-only check passed" in run.output, run.output
