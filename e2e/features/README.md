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
| `edge_host_bootstrapped` | `reachy_mini: str` | Runs `scripts/edge_host_bootstrap.sh` there (installs once, then only validates) and checks the pinned reachy-mini version, the robot device and, on macOS, Reachy Edge.app |
| `code_synced_to_edge_host` | none | `git push`es this checkout's HEAD to the edge host over SSH, checks it out in `~/assistant-edge/src` and `uv sync`s only the edge packages |
| `start_reachy_daemon` | `ready_within_s = 120` | Starts the real reachy-mini daemon there WITH media (`python -m assistant_robot_reachy.daemon` from the synced checkout, on macOS inside Reachy Edge.app: API on 127.0.0.1, WebRTC signalling on 127.0.0.1:8443, no mDNS; no wake-up/sleep motion); fails if a daemon it did not start already answers |
| `reachy_daemon_running` | `ready_within_s = 120` | Reuses a daemon that already answers there, else starts one as above; teardown stops only one it started (the `reachy_daemon` pytest fixture does the same) |
| `responsible_process_is_app` | `process = daemon` (or `client:<id>`) | macOS: every process of it has Reachy Edge.app as its responsible process (the owner of the camera and microphone permission); no-op elsewhere |
| `daemon_ports_on_loopback` | `listening: list[int]?` (default 8000, 8443) | `lsof` of the daemon's process group: every socket is on 127.0.0.1 / [::1] at both ends (no `*`, no LAN address, no mDNS 5353) and it LISTENs on 127.0.0.1 at each port |
| `daemon_status_is` | `state = running`, `version: str?` | `GET /api/daemon/status` (through the tunnel): state, backend ready, no error |
| `robot_state_read` | `control_mode: str?` | `GET /api/state/full`: head pose, body yaw, both antennas (and the motor mode) |
| `stop_reachy_daemon` | none | SIGTERM; the daemon reports a clean stop and stops answering |
| `edge_host_clean` | none | Stops this scenario's processes on the edge host; nothing from `~/assistant-edge` may still run there |

### The edge agent and the robot (`steps/edge.py`)

The real edge agent (`python -m assistant_edge --body console|reachy`) is a link client: its
process is `client:<id>`, so the link steps above (`server_sends`, `server_receives`,
`kill_process`, `restart_process`, `client_reconnects_within`, ...) work on it too. `body` is
`console` (prints what it would do; its speaker is a real-time playback clock with no sound
device) or `reachy` (the real robot through the reachy-mini SDK; runs where the robot is,
`where: edge_host`, on macOS inside Reachy Edge.app, and needs the daemon). Robot moves are measured from the daemon's
`/api/state/full`, sampled while the robot moves.

Robot safety: nothing moves unless the robot is detected. Expressions are Pollen's recorded
emotion moves (dataset `pollen-robotics/reachy-mini-emotions-library`, cached on the edge host
by `scripts/edge_host_bootstrap.sh`, played with `ReachyMini.play_move`). A robot at rest
(motors off) follows the SDK's standard pattern: `wake_up()` -> the move from neutral -> back
to neutral -> `goto_sleep()` -> motors off. Pollen's own tested motions (`wake_up()`,
`goto_sleep()`, the recorded moves) are the one exception to our limits; every move WE author
stays small and slow (head at most 10 degrees, antennas at most 20). On any error the arbiter
ends with `goto_sleep()` and the motors off. `robot_plays_emotion` measures with a sampler
running next to the daemon (20 Hz, on the robot's machine, so no SSH tunnel or PC load is in
the path), over exactly the samples taken while the move played: the body's MOTION line gives
the move's start and end on that machine's monotonic clock. Sampling below 8 Hz or with a gap
over 0.4 s fails as "the measurement is starved".

| Step | Arguments | Does |
|---|---|---|
| `start_edge_agent` | `id`, `body: console \| reachy = console`, `where: pc \| edge_host = pc`, `wait = true`, `energy_trigger_dbfs: float?`, `within_s = 60` | Starts the edge agent; waits for its BODY line and welcome unless `wait: false` |
| `edge_body_is` | `client`, `aec: none \| sw \| hw?`, `camera: bool?`, `expressions: list[str]?` | The body's announced capabilities (and the hello the server got); `aec: hw` also checks the XVF3800 report (board found, one far-end reference) |
| `edge_types` | `client`, `text` | Types a line into the agent (`/ptt down`, `/mute`, ... or text, sent as text.input) |
| `press_push_to_talk` / `release_push_to_talk` | `client` | `/ptt down` opens a mic window (MIC-OPEN); `/ptt up` closes it (MIC-CLOSE) |
| `edge_answers` | `client`, `type`, `fields: map?`, `ok = true`, `error_contains: str?`, `within_s = 10` | The server sends a request; the edge's `result` for that id has `ok` (and the error text) |
| `edge_output_clean` | `client` | No traceback, CONSOLE-ERROR, UNHANDLED or BODY-ERROR line from the agent |
| `edge_agent_fails` | `client`, `contains: str`, `within_s = 60` | The agent exited non-zero, said `contains` and never connected |
| `client_stayed_connected` | `client` | One welcome, no CLOSED, no RETRY: the link never dropped |
| `server_streams_speech` | `client`, `stream: int`, `clip: str`, `rate = 24000`, `text: str?` | The server console's `stream` command: speak.begin, 20 ms 0x02 frames, speak.end; `clip` is `tone:<hz>:<seconds>` or a 16-bit mono WAV path |
| `playback_reported` | `client`, `stream`, `state: started \| progress \| done \| flushed`, `min_played_ms: int?`, `max_played_ms: int?`, `within_s = 10` | The server got the edge's playback clock in that state (`progress`: the first with at least `min_played_ms`) |
| `playback_paced` | `client`, `stream`, `ms: int` | started -> done took 90 % of `ms` to `ms` + 1.5 s, and done reports `ms` (±5 %) played |
| `playback_stops_within` | `client`, `stream`, `ms: float`, `local = true` | The edge's FLUSHED line (local on barge-in, else the brain's flush) took at most `ms`, and the server got playback{flushed} |
| `barge_in_reported` | `client`, `stream`, `min_played_ms = 0`, `max_played_ms: int?`, `within_s = 5` | The server got vad{start, barge_in, stream_id, played_ms} |
| `uplink_audio_live` | `client`, `min_frames: int`, `above_dbfs = -100`, `within_s = 10` | At least `min_frames` mic frames reached the server; the loudest is above `above_dbfs` and the level varies (not digital silence) |
| `uplink_carries_no_audio` | `client`, `seconds: float` | No mic frame from the edge reaches the server for `seconds` |
| `robot_plays_emotion` | `client`, `emotion`, `move`, `min_head_deg = 0`, `min_antenna_deg = 0` | express{emotion}; the result is ok, the body's MOTION line names Pollen's `move` played to its end, and while it played the head turned at least `min_head_deg` from its pose at the start (the rotation angle, any axis) and an antenna at least `min_antenna_deg` |
| `robot_back_at_rest` | `head_deg = 2`, `antenna_deg = 5` | After the move (and `goto_sleep()` for a robot that was at rest) the head (yaw relative to the body) and antennas are back at their start and the motor mode is what it was |
| `robot_camera_frame` | `client`, `slot = 1`, `max_side = 640`, `min_bytes = 2000` | snapshot; the server reassembles a whole JPEG on `slot` that fits `max_side` and matches the result |
| `edge_body_lost` | `client`, `within_s = 10` | The daemon went away: BODY-ERROR from the agent, error{body_unavailable} at the server |
| `edge_body_recovers` | `client`, `within_s = 30` | The agent reconnected its body on its own (BODY-OK) |

Later tasks add `start` (brain) and the model steps.
