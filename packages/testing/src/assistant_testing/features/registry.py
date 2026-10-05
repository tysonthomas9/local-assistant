"""The step registry: `@step("name")` turns an async function into a feature-file step.

    @step("start_edge")
    async def start_edge(ctx: ScenarioContext, name: str, body: str = "console") -> None: ...

The parameters after `ctx` become a pydantic model, so step arguments are type-checked when
the feature file is collected. In YAML a step is a one-key mapping: `- start_edge: {name: desk}`.
A scalar argument fills the first parameter (`- start: brain`); no argument means no parameters.
"""

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from assistant_testing.features.context import ScenarioContext

StepFn = Callable[..., Awaitable[None]]


class StepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


@dataclass(frozen=True)
class StepDef:
    name: str
    fn: StepFn
    args_model: type[StepArgs]
    params: tuple[str, ...]
    doc: str

    def bind(self, raw: Any) -> dict[str, Any]:
        """Validate raw YAML arguments; return keyword arguments for `fn`."""
        if raw is None:
            data: Mapping[str, Any] = {}
        elif isinstance(raw, Mapping):
            data = raw
        elif self.params:
            data = {self.params[0]: raw}
        else:
            raise ValueError(f"step {self.name!r} takes no arguments")
        model = self.args_model.model_validate(dict(data))
        return {name: getattr(model, name) for name in self.params}


REGISTRY: dict[str, StepDef] = {}


def step(name: str) -> Callable[[StepFn], StepFn]:
    def register(fn: StepFn) -> StepFn:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"step {name!r} must be an async function")
        if name in REGISTRY:
            raise ValueError(f"step {name!r} is registered twice")
        params = list(inspect.signature(fn, eval_str=True).parameters.values())
        if not params or params[0].annotation is not ScenarioContext:
            raise TypeError(f"step {name!r}: the first parameter must be `ctx: ScenarioContext`")
        fields: dict[str, Any] = {}
        for param in params[1:]:
            if param.annotation is inspect.Parameter.empty:
                raise TypeError(f"step {name!r}: parameter {param.name!r} needs a type")
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[param.name] = (param.annotation, default)
        model = create_model(f"{name}_args", __base__=StepArgs, **fields)
        REGISTRY[name] = StepDef(
            name=name,
            fn=fn,
            args_model=model,
            params=tuple(p.name for p in params[1:]),
            doc=inspect.cleandoc(fn.__doc__ or ""),
        )
        return fn

    return register


def describe(name: str, raw: Any, width: int = 110) -> str:
    """One-line text of a step as written in the feature file."""
    if raw is None:
        text = name
    elif isinstance(raw, Mapping):
        parts = [f"{k}={_short(v)}" for k, v in raw.items()]
        text = f"{name} " + " ".join(parts)
    else:
        text = f"{name}: {_short(raw)}"
    return text if len(text) <= width else text[: width - 1] + "…"


def _short(value: Any) -> str:
    if isinstance(value, str):
        return value if value and " " not in value else json.dumps(value)
    return json.dumps(value, default=str, separators=(",", ":"))
