"""Load and validate feature files; every error names the file and line."""

import dataclasses
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assistant_testing.features.registry import REGISTRY, StepDef, describe

Tier = Literal["core", "sim", "hw", "models"]
TierSpec = Tier | Annotated[list[Tier], Field(min_length=1)]

REAL_BODIES = ("reachy", "console")
"""The only body types e2e may use (E2E uses only real devices and the real stack)."""

RealBody = Literal["reachy", "console"]
"""Type for step parameters that take a body."""


class FeatureError(Exception):
    """A feature file is invalid. The message starts with `path:line:`."""


class _ScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    steps: list[dict[str, Any] | str] = Field(min_length=1)
    examples: list[dict[str, Any]] | None = Field(default=None, min_length=1)
    """Scenario outline: one scenario per example; `<key>` in the name and steps is replaced."""
    tier: TierSpec | None = None
    """This scenario's tier(s) instead of the feature's, e.g. `hw` for a scenario of a
    `[sim, hw]` feature that the simulated robot cannot run."""


_PLACEHOLDER = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>")


def _fill(value: Any, example: dict[str, Any], unknown: set[str]) -> Any:
    """Replace `<key>` placeholders. A string that is exactly `<key>` takes the raw value."""
    if isinstance(value, str):
        whole = _PLACEHOLDER.fullmatch(value)
        if whole is not None and whole.group(1) in example:
            return example[whole.group(1)]

        def one(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in example:
                unknown.add(key)
                return match.group(0)
            item = example[key]
            return item if isinstance(item, str) else json.dumps(item, default=str)

        return _PLACEHOLDER.sub(one, value)
    if isinstance(value, dict):
        return {_fill(k, example, unknown): _fill(v, example, unknown) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(item, example, unknown) for item in value]
    return value


def _expand(
    spec: "_ScenarioSpec", index: int
) -> list[tuple[str, list[dict[str, Any] | str], list[int | str]]]:
    """(name, steps, yaml location) of every scenario an entry produces (outlines: several)."""
    if spec.examples is None:
        return [(spec.name, spec.steps, ["scenarios", index])]
    expanded: list[tuple[str, list[dict[str, Any] | str], list[int | str]]] = []
    for e_idx, example in enumerate(spec.examples):
        unknown: set[str] = set()
        name = _fill(spec.name, example, unknown)
        steps = _fill(spec.steps, example, unknown)
        if unknown:
            raise ValueError(
                f"examples[{e_idx}] has no value for placeholder(s): {', '.join(sorted(unknown))}"
            )
        expanded.append((name, steps, ["scenarios", index, "examples", e_idx]))
    return expanded


class _FeatureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature: str = Field(min_length=1)
    tier: TierSpec
    """One tier, or a list such as `[hw, models]`: the scenarios need all of them. `sim`
    always comes with `hw` (`[sim, hw]`): the scenario runs on the simulated robot AND on the
    physical one, as two items."""
    description: str = Field(min_length=1)
    scenarios: list[_ScenarioSpec] = Field(min_length=1)


@dataclass(frozen=True)
class BoundStep:
    name: str
    text: str
    line: int
    definition: StepDef
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class Scenario:
    name: str
    line: int
    steps: tuple[BoundStep, ...]
    tiers: tuple[Tier, ...] = ()
    """Its tiers (the feature's unless the scenario sets `tier`)."""


@dataclass(frozen=True)
class Feature:
    path: Path
    name: str
    tiers: tuple[Tier, ...]
    """Every tier this feature needs; each becomes a pytest marker."""
    description: str
    scenarios: tuple[Scenario, ...]

    @property
    def tier(self) -> str:
        """The tier as written: `core`, or `[hw, models]` for several."""
        return self.tiers[0] if len(self.tiers) == 1 else f"[{', '.join(self.tiers)}]"


def _line_of(node: yaml.Node | None, loc: Sequence[int | str]) -> int:
    """1-based line of the YAML node at `loc` (or of its deepest existing ancestor)."""
    if node is None:
        return 1
    line = node.start_mark.line + 1
    for key in loc:
        child: yaml.Node | None = None
        if isinstance(node, yaml.MappingNode):
            for key_node, value_node in node.value:
                if key_node.value == key:
                    child = value_node
                    break
        elif (
            isinstance(node, yaml.SequenceNode)
            and isinstance(key, int)
            and 0 <= key < len(node.value)
        ):
            child = node.value[key]
        if child is None:
            break
        node = child
        line = node.start_mark.line + 1
    return line


def _bad_bodies(value: Any, loc: list[int | str]) -> list[tuple[list[int | str], Any]]:
    """Every `body:` value anywhere in a step's arguments that is not a real body type."""
    found: list[tuple[list[int | str], Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "body" and not isinstance(child, dict | list):
                if child not in REAL_BODIES:
                    found.append(([*loc, "body"], child))
            else:
                found.extend(_bad_bodies(child, [*loc, str(key)]))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_bad_bodies(child, [*loc, index]))
    return found


def load_feature(path: Path) -> Feature:
    text = path.read_text()
    try:
        root = yaml.compose(text)
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 1 if mark is not None else 1
        raise FeatureError(f"{path}:{line}: invalid YAML: {exc}") from None

    def fail(loc: Sequence[int | str], message: str) -> FeatureError:
        return FeatureError(f"{path}:{_line_of(root, loc)}: {message}")

    def tiers_of(tier: Tier | list[Tier], loc: Sequence[int | str]) -> tuple[Tier, ...]:
        listed: list[Tier] = [tier] if isinstance(tier, str) else tier
        tiers = tuple(dict.fromkeys(listed))
        if "sim" in tiers and "hw" not in tiers:
            raise fail(
                loc,
                "tier sim needs hw too ([sim, hw]): the simulated robot is for iteration, "
                "every robot scenario also runs on the physical robot",
            )
        return tiers

    try:
        spec = _FeatureSpec.model_validate(data)
    except ValidationError as exc:
        err = exc.errors()[0]
        loc = [p for p in err["loc"] if isinstance(p, int | str)]
        where = ".".join(str(p) for p in loc) or "(top level)"
        raise fail(loc, f"{where}: {err['msg']}") from None

    feature_tiers = tiers_of(spec.tier, ["tier"])
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for s_idx, scenario_spec in enumerate(spec.scenarios):
        tiers = (
            feature_tiers
            if scenario_spec.tier is None
            else tiers_of(scenario_spec.tier, ["scenarios", s_idx, "tier"])
        )
        try:
            expanded = _expand(scenario_spec, s_idx)
        except ValueError as exc:
            raise fail(["scenarios", s_idx, "examples"], str(exc)) from None
        for name_, raw_steps, where in expanded:
            if name_ in seen:
                raise fail([*where, "name"], f"duplicate scenario {name_!r}")
            seen.add(name_)
            bound = _bind_scenario(fail, root, s_idx, name_, raw_steps, where)
            scenarios.append(dataclasses.replace(bound, tiers=tiers))
    return Feature(
        path=path,
        name=spec.feature,
        tiers=feature_tiers,
        description=spec.description,
        scenarios=tuple(scenarios),
    )


def _bind_scenario(
    fail: Callable[[Sequence[int | str], str], FeatureError],
    root: yaml.Node | None,
    s_idx: int,
    scenario_name: str,
    raw_steps: list[dict[str, Any] | str],
    where: list[int | str],
) -> Scenario:
    """Bind the steps of one (possibly outline-expanded) scenario to their step functions."""
    steps: list[BoundStep] = []
    for st_idx, raw_step in enumerate(raw_steps):
        loc: list[int | str] = ["scenarios", s_idx, "steps", st_idx]
        if isinstance(raw_step, str):
            name, raw_args = raw_step, None
        elif len(raw_step) == 1:
            ((name, raw_args),) = raw_step.items()
        else:
            raise fail(loc, "a step is a name, or a mapping with exactly one key (the name)")
        for body_loc, body in _bad_bodies(raw_args, [*loc, name]):
            raise fail(
                body_loc,
                f"step {name!r}: body {body!r} is not a real body type "
                f"(allowed: {', '.join(REAL_BODIES)}); e2e uses only real devices",
            )
        definition = REGISTRY.get(name)
        if definition is None:
            known = ", ".join(sorted(REGISTRY))
            raise fail(loc, f"unknown step {name!r}; known steps: {known}")
        try:
            kwargs = definition.bind(raw_args)
        except (ValidationError, ValueError) as exc:
            detail = (
                "; ".join(
                    f"{'.'.join(str(p) for p in e['loc']) or name}: {e['msg']}"
                    for e in exc.errors()
                )
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            raise fail([*loc, name], f"step {name!r}: {detail}") from None
        steps.append(
            BoundStep(
                name=name,
                text=describe(name, raw_args),
                line=_line_of(root, loc),
                definition=definition,
                kwargs=kwargs,
            )
        )
    return Scenario(name=scenario_name, line=_line_of(root, where), steps=tuple(steps))
