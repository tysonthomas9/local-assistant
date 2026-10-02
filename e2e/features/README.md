E2E uses only real devices and the real stack: the real Reachy Mini, real brain and edge processes over real EdgeLink, and real models. No fakes, mocks, stubs, simulators, recorded replies or monkeypatching. Faults are injected for real.

# E2E feature files

Each YAML file here is one feature. Every scenario in it becomes one pytest item, run by the
plugin in `packages/testing` (`assistant_testing.features`). Scenarios drive **real** code
and processes. A scenario that needs the robot is tier `hw`, and one that needs the GPUs and
model servers is tier `models`.

The rule is enforced. `scripts/gate.sh` runs a "real-only check"
(`python -m assistant_testing.real_only`) before the e2e stages, and the feature
`core/real_only.yaml` runs the same check. It fails on imports of the mock libraries, on
monkeypatching, and on any identifier that starts with Fake, Mock, Stub or Dummy followed by a
capital letter. It prints the offending file and line. It scans `e2e/` and the runner and step
modules; the exact rules are in `packages/testing/src/assistant_testing/real_only.py`.
Wherever a step takes a `body` argument, it must be a real body type: `reachy` or `console`.

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

**Scenario outlines.** A scenario with `examples:` runs once per example. Every `<key>` in
its name and steps is replaced by that example's value; a string that is exactly `<key>`
takes the raw value (a mapping or list), so one outline can send different message fields.
The expanded names must be unique.

```yaml
  - name: "edge -> brain: <type>"
    examples:
      - {type: wake, fields: {word: hey jarvis, score: 0.8}}
      - {type: text.input, fields: {text: what time is it}}
    steps:
      - start_link_server
      - start_link_client: desk
      - client_sends: {client: desk, type: <type>, fields: <fields>}
      - server_receives: {client: desk, type: <type>, fields: <fields>}
```

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
| `expect_assistant` | `id: str`, `path: str`, `equals: any` | Compares a dotted path, e.g. `wake_words.0.spoken`, of a loaded assistant |
| `real_only_check` | none | Runs the real-only check over this checkout in a subprocess |

### EdgeLink contracts (`steps/contracts.py`)

| Step | Arguments | Does |
|---|---|---|
| `roundtrip_message` | `type: str`, `fields: map?` | JSON-encodes a message with a full envelope, decodes it through the union, and checks nothing changed |
| `expect_message_refused` | `wire: map` | Decoding this JSON must fail validation |
| `expect_all_message_types_covered` | none | All 23 message types were round-tripped in this scenario |
| `expect_message_type_order` | `names: list[str]` | The union defines exactly these types, in this order |
| `roundtrip_frame` | `kind: str` (`mic_pcm`, `out_pcm`, `jpeg_chunk`, `sound_clip`, `opus`), `payload_bytes: int = 640`, `stream`, `seq`, `capture_ts_us`, `opus_negotiated: bool = false` | Encodes and decodes a binary frame |
| `expect_frame_refused` | `kind: str`, `reason: not_negotiated \| malformed`, `payload_bytes: int = 4` | Encoding and decoding the frame must both fail |
| `expect_all_frame_kinds_covered` | none | Kinds 0x01-0x05 were all round-tripped in this scenario |

### EdgeLink over a real link (`steps/link.py`)

A real link server console (`python -m assistant_link.server --console`) and real client
consoles (`python -m assistant_link.client --console`) on a free 127.0.0.1 port, one process
each. Steps type commands into their stdin and read their output. Every expectation
consumes the output line it matched. `client` is a device id; `process` is `server` or a
device id. Messages are checked by containment: the listed `fields` must be in the message.

| Step | Arguments | Does |
|---|---|---|
| `start_link_server` | `accept_opus: bool = false`, `token: str` | "start the link server console" |
| `start_link_client` | `id: str`, `token: str?`, `proto: str?`, `opus: bool = false`, `speak_text: bool = false`, `wait: bool = true`, `where: pc \| edge_host = pc` | "start a link client "<id>" [with token] [with proto]"; waits for its welcome unless `wait: false`. `where: edge_host` runs it on the robot's machine (see below) |
| `server_connected` | `client: str`, `fields: map?`, `within_s = 5` | The server accepted the client's hello (containing `fields`) |
| `client_sends` | `client`, `type`, `fields: map?` | "client "<id>" sends <type> <fields>" |
| `server_receives` | `type`, `client: str?`, `fields: map?`, `within_s = 5` | "the server receives <type>" |
| `server_sends` | `client`, `type`, `fields: map?` | "the server sends <type> <fields>" |
| `client_receives` | `client`, `type`, `fields: map?`, `within_s = 5` | "client "<id>" receives <type>" (`welcome` too) |
| `client_sends_raw` / `server_sends_raw` | `client`, `message: map?` or `text: str` | Puts unchecked text on the wire (tests the receiver's checks) |
| `sender_refuses_locally` | `client`, `sender: edge \| brain`, `type`, `fields: map?` | The sender's own link refuses a wrong-direction type |
| `server_refuses` | `code: str`, `client: str?`, `within_s = 5` | The server refused a connection or item (`auth`, `version_mismatch`, `wrong_direction`, `frame_refused`, `bad_message`) |
| `client_refuses` | `client`, `code: str`, `within_s = 5` | The client refused an item the server sent |
| `client_sends_frame` | `client`, `kind`, `bytes = 640`, `stream = 1`, `seq = 1`, `unchecked = false`, `expect: sent \| refused_locally` | "client "<id>" sends frame <kind>" (`kind`: `mic_pcm`, `out_pcm`, `jpeg_chunk`, `sound_clip`, `opus` or `"0x01"`) |
| `server_receives_frame` | `client`, `kind`, `bytes: int?`, `within_s = 5` | "the server receives frame <kind>", with the payload CRC the client sent |
| `server_sends_frame` | `client`, `kind`, ... as `client_sends_frame` | The server sends a frame to the client |
| `client_receives_frame` | `client`, `kind`, `bytes: int?`, `within_s = 5` | The client received the frame, with the payload CRC the server sent |
| `capability_negotiated` | `client`, `capability: opus \| speak_text`, `negotiated: bool = true` | "capability "<cap>" is negotiated" (both ends agree for `opus`) |
| `connection_closed_with_code` | `client`, `code: int`, `within_s = 5` | "the connection is closed with code <n>"; for 4001/4003 the client also gives up and exits 2 |
| `server_disconnects` | `client`, `within_s = 5`, `code: int?`, `not_before_s = 0`, `stalled: bool?` | The server dropped the client's session (`stalled`: by the write-stall watchdog) |
| `client_disconnects` | `client`, `within_s = 5`, `stalled: bool?` | The client's connection dropped; it keeps running and retries |
| `server_floods` | `client`, `kind = out_pcm`, `count = 1000000`, `bytes = 1048576`, `stream = 1` | The server sends frames to the client as fast as the link takes them |
| `client_floods` | `client`, `kind = jpeg_chunk`, `count`, `bytes`, `stream` | The client floods frames to the server |
| `flood_stopped` | `process`, `within_s = 5` | The flood ended early because its link closed (the writes were backed up) |
| `peer_leaves_before_hello` | `stage: tcp \| upgraded = upgraded` | A real TCP peer connects (and, if `upgraded`, completes the WebSocket upgrade) and hangs up before hello |
| `server_output_clean` | none | The server printed no traceback or handler error |
| `kill_process` | `process`, `signal_name: KILL \| STOP \| CONT = KILL` | "kill the <process>" with a real signal (kill -9, freeze, thaw) |
| `restart_process` | `process` | "restart the <process>" with the same command line (the server keeps its port) |
| `client_reconnects_within` | `client`, `seconds: float` | "client "<id>" reconnects within <s> s": a new welcome and a new server session |
| `client_retries_with_backoff` | `client`, `min_retries = 1` | Every printed retry delay is 0.5 s doubling to 10 s, within ±20 % jitter |
| `wait` | `seconds: float` | Lets real time pass |

### The robot's machine (`steps/edge_host.py`)

The robot may be plugged into another machine, the **edge host**, reached by an SSH alias
(`[test.edge_host] ssh` in `config/assistant.toml`, or `ASSISTANT_EDGE_HOST`); a robot attached
to this PC is always used first. Processes started there run through SSH but behave like local
ones: their output streams back, steps type into them, `kill_process` signals reach the
remote process group, and teardown stops them. The daemon API (`ssh -L`) and EdgeLink
(`ssh -R`) stay on 127.0.0.1 at both ends. See `docs/robot-on-another-machine.md`.

| Step | Arguments | Does |
|---|---|---|
| `robot_host_found` | none | Picks the machine with the robot and prints it; fails (never skips) if neither this PC nor the edge host has its USB serial device |
| `edge_host_bootstrapped` | `reachy_mini: str` | Runs `scripts/edge_host_bootstrap.sh` there (installs once, then only validates) and checks the pinned reachy-mini version and the robot device |
| `code_synced_to_edge_host` | none | `git push`es this checkout's HEAD to the edge host over SSH, checks it out in `~/assistant-edge/src` and `uv sync`s only the edge packages |
| `start_reachy_daemon` | `ready_within_s = 90` | Starts the real reachy-mini daemon there (API on 127.0.0.1, no media, no wake-up/sleep motion); fails if a daemon it did not start already answers |
| `daemon_status_is` | `state = running`, `version: str?` | `GET /api/daemon/status` (through the tunnel): state, backend ready, no error |
| `robot_state_read` | `control_mode: str?` | `GET /api/state/full`: head pose, body yaw, both antennas (and the motor mode) |
| `stop_reachy_daemon` | none | SIGTERM; the daemon reports a clean stop and stops answering |
| `edge_host_clean` | none | Stops this scenario's processes on the edge host; nothing from `~/assistant-edge` may still run there |

Later tasks add `start` (brain), `start_edge`, `expect_message` and the robot and model steps.
