# E2E feature files

Each YAML file here is one feature. Every scenario in it becomes one pytest item, run by the
plugin in `packages/testing` (`assistant_testing.features`). Scenarios drive **real** code
and processes. There are no fakes: a scenario that needs the robot is tier `hw`, and one
that needs the GPUs and model servers is tier `models`.

```yaml
feature: EdgeLink handshake          # short name
tier: core                           # core | hw | models (becomes a pytest marker)
description: A device connects to the brain and is welcomed.
scenarios:
  - name: device says hello
    steps:
      - start: brain                 # scalar argument -> the step's first parameter
      - start_edge: {name: desk, body: reachy}
      - expect_message: {edge: desk, type: welcome, within_ms: 500}
```

A step is a one-key mapping: the step name and its arguments, given as a mapping, a single
scalar, or nothing. Step arguments are type-checked when the file is collected. Errors name
the file and line.

## Running

```bash
uv run pytest e2e -m core            # tier core (hosted CI)
uv run pytest e2e -m hw              # needs the robot
uv run pytest e2e -m models          # needs the GPUs and model servers
uv run pytest e2e --list-features    # list features, scenarios and steps; run nothing
scripts/gate.sh                      # the full gate (see the script header)
```

Each scenario prints its steps with ✓ (passed), ✗ (failed) or - (not reached). Every process a
scenario starts is stopped in teardown, even when a step fails.

## Step vocabulary

Steps are async functions registered with `@step("name")` in `assistant_testing/steps/`.
The parameters after `ctx` are the step's arguments.

### Workspace (`steps/core.py`)

| Step | Arguments | Does |
|---|---|---|
| `import_packages` | `packages: list[str]` (default: all 11) | Imports the packages in a fresh Python process and checks they come from this checkout |
| `load_config` | `profile: str?`, `env: {NAME: value}?` | Loads `config/` (+ profile). Only `env` is used as the environment |
| `expect_config` | `path: str`, `equals: any` | Compares a dotted path, e.g. `llm.impl`, of the last loaded config |
| `load_assistant` | `id: str` | Loads and validates `config/assistants/<id>.toml` |

### EdgeLink contracts (`steps/contracts.py`)

| Step | Arguments | Does |
|---|---|---|
| `roundtrip_message` | `type: str`, `fields: map?` | JSON-encodes a message with a full envelope, decodes it through the union, and checks nothing changed |
| `expect_message_refused` | `wire: map` | Decoding this JSON must fail validation |
| `expect_all_message_types_covered` | none | All 23 message types were round-tripped in this scenario |
| `roundtrip_frame` | `kind: str` (`mic_pcm`, `out_pcm`, `jpeg_chunk`, `sound_clip`, `opus`), `payload_bytes: int = 640`, `stream`, `seq`, `capture_ts_us`, `opus_negotiated: bool = false` | Encodes and decodes a binary frame |
| `expect_frame_refused` | `kind: str`, `reason: not_negotiated \| malformed`, `payload_bytes: int = 4` | Encoding and decoding the frame must both fail |
| `expect_all_frame_kinds_covered` | none | Kinds 0x01-0x05 were all round-tripped in this scenario |

Later tasks add `start` (brain), `start_edge`, `expect_message` and the robot and model steps.
