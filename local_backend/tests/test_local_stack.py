"""End-to-end checks for the fully local Reachy Mini conversation stack.

Run with the speech-to-speech venv (it has openai, websockets, numpy, pytest):

    third_party/speech-to-speech/.venv/bin/python -m pytest local_backend/tests -v

Needs the daemon, Ollama and the speech-to-speech server running. The `realtime`
tests take the speech server's only session slot, so stop the conversation app
first or they are skipped. Latencies and levels are appended to
local_backend/logs/test_results.jsonl so runs can be compared over time.
"""

import asyncio
import base64
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import websockets
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
RESULTS = ROOT / "local_backend" / "logs" / "test_results.jsonl"

DAEMON = "http://127.0.0.1:8000"
OLLAMA = "http://127.0.0.1:11434"
REALTIME = "ws://127.0.0.1:8765/v1/realtime"
MODEL = os.environ.get("REACHY_LLM", "reachy-gemma4")
APP_PYTHON = ROOT / "reachy_mini_conversation_app" / ".venv" / "bin" / "python"

APP_REQUEST = json.loads((FIXTURES / "app_request.json").read_text())
CAMERA_JPEG = (FIXTURES / "camera_frame.jpg").read_bytes()

# Latency budgets (seconds, warm model). The hosted backend took 2.0-2.4 s to reply text
# and 4.4 s to describe a camera image, so these are "at least as good as hosted".
LLM_FIRST_TOKEN_MAX = 1.0          # median of warm calls
LLM_COLD_FIRST_TOKEN_MAX = 2.5     # the first call after other traffic re-processes the prompt
SESSION_FIRST_AUDIO_MAX = 2.5      # first turn of a session also processes the ~4k-char system prompt + 17 tools
REALTIME_FIRST_AUDIO_MAX = 1.5     # later turns
IMAGE_FIRST_AUDIO_MAX = 8.0
TOOL_RUNS = 8
TOOL_MIN_HITS = 6
VOICE_MIN_PEAK_DBFS = -14   # observed -1 to -9 dBFS; -14 still catches a broken/quiet TTS
GPU_MIN_HEADROOM_MIB = 1536


def record(test: str, **values) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "model": MODEL, "test": test, **values}) + "\n")


def http_json(url: str, data: dict | None = None) -> dict:
    cmd = ["curl", "-sf", url] + (["-d", json.dumps(data)] if data is not None else [])
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def has_emoji(text: str) -> bool:
    return any(ord(ch) >= 0x2600 for ch in text)


@pytest.fixture(scope="session")
def llm() -> OpenAI:
    client = OpenAI(base_url=f"{OLLAMA}/v1", api_key="ollama")
    # Load the model before any timed call.
    client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}],
                                   max_tokens=1, extra_body={"reasoning_effort": "none"})
    return client


def chat(llm: OpenAI, user_content, history=None, **kwargs):
    messages = [{"role": "system", "content": APP_REQUEST["system"]}, *(history or []),
                {"role": "user", "content": user_content}]
    t0 = time.time()
    r = llm.chat.completions.create(model=MODEL, messages=messages, tools=APP_REQUEST["tools"],
                                    tool_choice="auto", max_tokens=300,
                                    extra_body={"reasoning_effort": "none"}, **kwargs)
    return r.choices[0].message, time.time() - t0


# --- Configuration -------------------------------------------------------------------------

def test_profile_is_offline_only():
    profile = (ROOT / "local_backend/profiles/local_reachy/profile.md").read_text()
    assert "pollen_robotics_" not in profile, "profile still enables a remote Hugging Face Space tool"
    assert '"get_time"' in profile


def test_captured_tool_list_has_no_remote_tools():
    names = [t["function"]["name"] for t in APP_REQUEST["tools"]]
    assert not [n for n in names if "__" in n], names
    assert {"dance", "move_head", "play_emotion", "camera", "get_time", "volume_control"} <= set(names)


def test_get_time_tool_runs():
    code = (
        "import asyncio, importlib.util, json;"
        f"spec = importlib.util.spec_from_file_location('get_time', '{ROOT}/local_backend/tools/get_time.py');"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "print(json.dumps([asyncio.run(m.GetTime()(None)), asyncio.run(m.GetTime()(None, timezone='Asia/Tokyo')),"
        " asyncio.run(m.GetTime()(None, timezone='Not/AZone'))]))"
    )
    out = subprocess.run([str(APP_PYTHON), "-c", code], capture_output=True, text=True, check=True).stdout
    local, tokyo, bad = json.loads(out.strip().splitlines()[-1])
    assert re.fullmatch(r"\d\d:\d\d", local["time"]) and local["date"]
    assert tokyo["timezone"] == "Asia/Tokyo"
    assert "error" in bad


def test_emotions_available_offline():
    code = ("from reachy_mini.motion.recorded_move import RecordedMoves;"
            "print(len(RecordedMoves('pollen-robotics/reachy-mini-emotions-library').list_moves()))")
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    out = subprocess.run([str(APP_PYTHON), "-c", code], capture_output=True, text=True, env=env, check=True).stdout
    assert int(out.strip().splitlines()[-1]) >= 80


def test_asoundrc_shares_the_robot_card():
    text = (Path.home() / ".asoundrc").read_text()
    assert "pcm.reachymini_audio_sink" in text and "pcm.reachymini_audio_src" in text
    assert "pcm.!default" not in text, "~/.asoundrc must not hijack the system default sound device"


def test_known_phone_home_paths_are_disabled():
    # onnxruntime 1.30 (speech server, Smart Turn model) ships Linux telemetry to
    # mobile.events.data.microsoft.com unless ORT_DISABLE_TELEMETRY is set.
    assert "ORT_DISABLE_TELEMETRY=1" in (ROOT / "local_backend/start_local_backend.sh").read_text()
    # speech-to-speech checks for the NLTK tagger under tokenizers/ (it lives under taggers/), so
    # without this link it re-downloads the NLTK index from raw.githubusercontent.com on every start.
    link = Path.home() / "nltk_data/tokenizers/averaged_perceptron_tagger_eng"
    assert link.exists(), f"missing {link} -> ../taggers/averaged_perceptron_tagger_eng"


def test_daemon_models_cached_and_launcher_offline():
    # Head tracking downloads the YuNet face model lazily on first use; the daemon also preloads the
    # emotions and dances datasets at startup. All must be cached for HF_HUB_OFFLINE=1.
    hub = Path.home() / ".cache/huggingface/hub"
    for repo in ("models--pollen-robotics--face_detection_yunet_2026may",
                 "datasets--pollen-robotics--reachy-mini-emotions-library",
                 "datasets--pollen-robotics--reachy-mini-dances-library"):
        assert (hub / repo).is_dir(), f"not cached: {repo} (run local_backend/cache_models.sh)"
    launcher = (ROOT / "start_daemon.sh").read_text()
    assert "export HF_HUB_OFFLINE=1" in launcher
    assert "--dataset-update-interval 0" in launcher
    assert "local_backend/run_daemon.py" in launcher


def test_ui_and_signalling_not_exposed_to_lan():
    listening = subprocess.run(["ss", "-ltnH"], capture_output=True, text=True, check=True).stdout
    for port, what in ((7860, "app web UI (run_app.py)"), (8443, "daemon WebRTC signalling (run_daemon.py)")):
        addrs = [l.split()[3] for l in listening.splitlines() if l.split()[3].endswith(f":{port}")]
        if not addrs:
            continue  # not running
        assert all(a.startswith(("127.0.0.1:", "[::1]:")) for a in addrs), f"{what} listens on {addrs}"


# --- Services ------------------------------------------------------------------------------

def test_daemon_running_with_media():
    status = http_json(f"{DAEMON}/api/daemon/status")
    assert status["state"] == "running", status.get("error")
    assert status["backend_status"]["control_loop_stats"]["nb_error"] == 0
    assert status["camera_specs_name"], "daemon found no camera"
    assert http_json(f"{DAEMON}/api/media/status")["available"] is True


def test_ollama_model_config():
    show = http_json(f"{OLLAMA}/api/show", {"model": MODEL})
    assert {"vision", "tools"} <= set(show["capabilities"])
    assert re.search(r"num_ctx\s+32768", show.get("parameters", "")), "expected the 32k-context variant"


# --- LLM behaviour (direct to Ollama, with the app's real system prompt and tools) ---------

def test_llm_thinking_off_and_fast(llm):
    firsts = []
    for _ in range(3):
        t0 = time.time(); first = None; reasoning = ""; text = ""
        for ch in llm.chat.completions.create(model=MODEL, stream=True, max_tokens=60,
                                              messages=[{"role": "user", "content": "Say hello in five words."}],
                                              extra_body={"reasoning_effort": "none"}):
            d = ch.choices[0].delta if ch.choices else None
            if d is None:
                continue
            reasoning += getattr(d, "reasoning", None) or ""
            text += d.content or ""
            if d.content and first is None:
                first = time.time() - t0
        # Ollama's OpenAI endpoint streams thinking as a separate `reasoning` delta; some templates
        # leak it into the content as <think>…</think> instead. Check both.
        assert not reasoning, "model is still thinking despite reasoning_effort=none"
        assert "<think>" not in (text or ""), f"thinking leaked into the answer: {text[:80]!r}"
        firsts.append(first)
    record("llm_first_token", seconds=[round(f, 3) for f in firsts])
    assert sorted(firsts)[1] < LLM_FIRST_TOKEN_MAX, firsts
    assert max(firsts) < LLM_COLD_FIRST_TOKEN_MAX, firsts


TOOL_CASES = [
    ("Can you do a dance for me?", "dance"),
    ("Look to your left.", "move_head"),
    ("Show me that you're happy!", "play_emotion"),
    ("What time is it?", "get_time"),
    ("What do you see right now?", "camera"),
    ("Turn your speaker volume up, please.", "volume_control"),
]


@pytest.mark.parametrize("prompt,tool", TOOL_CASES, ids=[t for _, t in TOOL_CASES])
def test_llm_calls_the_right_tool(llm, prompt, tool):
    results = []
    for _ in range(TOOL_RUNS):
        msg, dt = chat(llm, prompt)
        results.append(([tc.function.name for tc in (msg.tool_calls or [])], msg.content or "", round(dt, 2)))
    hits = sum(tool in names for names, _, _ in results)
    record("tool_call", tool=tool, hits=hits, runs=TOOL_RUNS, seconds=[r[2] for r in results])
    assert hits >= TOOL_MIN_HITS, f"{tool}: {hits}/{TOOL_RUNS} -> {results}"


def test_llm_speaks_plainly(llm):
    replies = [chat(llm, "Tell me a fun fact about robots.")[0].content or "" for _ in range(4)]
    bad = [r for r in replies if has_emoji(r) or re.search(r"\*[^*]+\*|\[[^\]]+\]|^#|^\s*[-*] ", r, re.M)]
    assert len(bad) <= 1, bad


def test_llm_admits_it_is_offline(llm):
    msg, _ = chat(llm, "What's the weather like in Paris today?")
    assert not msg.tool_calls
    assert not re.search(r"sunny|rain(y|ing)?\b|overcast|°|degrees|forecast (is|for)", msg.content or "", re.I), msg.content
    assert re.search(r"offline|can't|cannot|don't have|unable|no access|not able", msg.content or "", re.I), msg.content


def test_llm_does_not_promise_reminders(llm):
    # qwen3.6 once replied "Sure! I'll remind you in ten minutes." with no tool that can do that.
    msg, _ = chat(llm, "Set a reminder for ten minutes from now.")
    assert not re.search(r"\bI('ll| will) remind you\b", msg.content or "", re.I), msg.content


def test_llm_describes_camera_image(llm):
    # Same shape as the app: the model calls `camera`, the tool returns {"image_attached": true},
    # then the frame arrives as the next user message.
    img = "data:image/jpeg;base64," + base64.b64encode(CAMERA_JPEG).decode()
    history = [
        {"role": "user", "content": "What do you see?"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "id": "call_cam",
            "function": {"name": "camera", "arguments": '{"question": "What do you see?"}'}}]},
        {"role": "tool", "tool_call_id": "call_cam", "content": '{"image_attached": true}'},
    ]
    msg, dt = chat(llm, [{"type": "image_url", "image_url": {"url": img}}], history=history)
    record("llm_image", seconds=round(dt, 2))
    # The frame shows a living room: sofa, round glass/gold coffee table, laptop, person under a blue blanket.
    assert re.search(r"sofa|couch", msg.content or "", re.I), msg.content
    assert re.search(r"table|blanket|laptop|living room", msg.content or "", re.I), msg.content


# --- Realtime server (the path the conversation app uses) ----------------------------------

async def realtime_turns():
    try:
        ws = await websockets.connect(REALTIME, max_size=None)
        first = json.loads(await asyncio.wait_for(ws.recv(), 10))
    except (OSError, websockets.exceptions.ConnectionClosed) as e:
        return f"cannot connect: {e}"
    if first.get("type") != "session.created":
        return f"unexpected first event {first}"
    try:
        tools = [{"type": "function", **t["function"]} for t in APP_REQUEST["tools"]]
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime",
                      "instructions": APP_REQUEST["system"], "tools": tools, "audio": {"output": {"voice": "Aiden"}}}}))

        async def turn(content):
            await ws.send(json.dumps({"type": "conversation.item.create",
                                      "item": {"type": "message", "role": "user", "content": content}}))
            t0 = time.time()
            await ws.send(json.dumps({"type": "response.create"}))
            out = {"first_audio": None, "pcm": b"", "text": "", "calls": []}
            while True:
                ev = json.loads(await asyncio.wait_for(ws.recv(), 60))
                t = ev["type"]
                if t.endswith("audio.delta"):
                    out["first_audio"] = out["first_audio"] or time.time() - t0
                    out["pcm"] += base64.b64decode(ev["delta"])
                elif t.endswith("audio_transcript.delta"):
                    out["text"] += ev["delta"]
                elif t == "response.function_call_arguments.done":
                    out["calls"].append((ev["name"], ev["call_id"]))
                elif t == "error":
                    raise RuntimeError(ev["error"])
                elif t == "response.done":
                    return out

        results = {"text": await turn([{"type": "input_text", "text": "Tell me a short joke."}])}
        results["tool"] = await turn([{"type": "input_text", "text": "Can you do a dance for me?"}])
        for _, call_id in results["tool"]["calls"]:
            await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                "type": "function_call_output", "call_id": call_id, "output": json.dumps({"status": "queued"})}}))
        img = "data:image/jpeg;base64," + base64.b64encode(CAMERA_JPEG).decode()
        results["camera"] = await turn([{"type": "input_text", "text": "What do you see right now?"}])
        for _, call_id in results["camera"]["calls"]:
            await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                "type": "function_call_output", "call_id": call_id, "output": json.dumps({"image_attached": True})}}))
        results["image"] = await turn([{"type": "input_image", "image_url": img}])
        return results
    except websockets.exceptions.ConnectionClosed as e:
        return f"connection closed: {e}"
    finally:
        await ws.close()


@pytest.fixture(scope="module")
def realtime():
    res = asyncio.run(realtime_turns())
    if isinstance(res, str):
        if "session slots" in res:
            pytest.skip("speech server slot in use: stop the conversation app to run realtime tests")
        pytest.fail(res)
    return res


def peak_dbfs(pcm: bytes) -> float:
    a = np.frombuffer(pcm, dtype=np.int16).astype(float)
    return 20 * np.log10(max(np.abs(a).max(), 1) / 32768)


@pytest.mark.realtime
def test_realtime_text_turn(realtime):
    t = realtime["text"]
    record("realtime_text", first_audio=round(t["first_audio"] or -1, 2), peak_dbfs=round(peak_dbfs(t["pcm"]), 1))
    assert t["pcm"] and t["text"].strip()
    assert t["first_audio"] < SESSION_FIRST_AUDIO_MAX
    assert peak_dbfs(t["pcm"]) > VOICE_MIN_PEAK_DBFS, "synthesised speech is unusually quiet"


@pytest.mark.realtime
def test_realtime_tool_turn(realtime):
    t = realtime["tool"]
    record("realtime_tool", first_audio=round(t["first_audio"] or -1, 2), calls=[c[0] for c in t["calls"]])
    assert "dance" in [c[0] for c in t["calls"]], t["text"]
    assert t["first_audio"] is None or t["first_audio"] < REALTIME_FIRST_AUDIO_MAX


@pytest.mark.realtime
def test_realtime_camera_tool_turn(realtime):
    assert "camera" in [c[0] for c in realtime["camera"]["calls"]], realtime["camera"]["text"]


@pytest.mark.realtime
def test_realtime_image_turn(realtime):
    t = realtime["image"]
    record("realtime_image", first_audio=round(t["first_audio"] or -1, 2))
    assert re.search(r"sofa|couch", t["text"], re.I), t["text"]
    assert t["first_audio"] < IMAGE_FIRST_AUDIO_MAX


# --- Locality and GPU ----------------------------------------------------------------------

STACK_PATTERNS = [
    ("daemon", r"^[^ ]*python[0-9.]* [^ ]*(reachy-mini-daemon|local_backend/run_daemon\.py)"),
    ("speech", r"^[^ ]*python[0-9.]* [^ ]*speech-to-speech serve"),
    ("app", r"^[^ ]*python[0-9.]* [^ ]*(reachy-mini-conversation-app|local_backend/run_app\.py)"),
]


def stack_pids() -> dict[str, list[str]]:
    pids = {}
    for label, pattern in STACK_PATTERNS:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
        if out:
            pids[label] = out
    return pids


def _external_stack_sockets(pids: set[str]) -> dict[tuple[str, str], str]:
    """One `ss` snapshot of established TCP sockets with a non-loopback peer that may be the stack's.

    Processes launched via `sg` have another group ID, so ss can't show their PID; their sockets
    still show this user's uid, so unattributed sockets of this user are candidates too.
    """
    ss = subprocess.run(["ss", "-tnepH", "state", "established"], capture_output=True, text=True, check=True).stdout
    found = {}
    for line in ss.splitlines():
        m = re.search(r"pid=(\d+)", line)
        ours = (m and m.group(1) in pids) or (not m and f"uid:{os.getuid()} " in line)
        if not ours:
            continue
        cols = line.split()
        local, peer = cols[2], cols[3]  # with a state filter, ss omits the state column
        if not re.match(r"(127\.|\[::1\]|\[::ffff:127\.)", peer):
            found[(local, peer)] = line
    return found


def test_no_external_connections():
    """Spot check: no *established* TCP connection from the stack to a non-loopback peer, right now.

    This sees a single moment (no UDP/DNS, nothing sub-second); the strace run documented in
    LOCAL_CONVERSATION.md is the real evidence. Unattributed sockets are ambiguous (another process
    of this user may have opened one between ss building its PID table and listing sockets), so a
    socket only counts if it is still unattributed-or-ours in three snapshots 0.3 s apart.
    """
    pids = {p for ps in stack_pids().values() for p in ps}
    assert pids, "no stack processes found"
    samples = []
    for i in range(3):
        if i:
            time.sleep(0.3)
        samples.append(_external_stack_sockets(pids))
    persistent = set(samples[0]).intersection(*samples[1:])
    external = [samples[-1][k] for k in persistent]
    assert not external, "stack has non-loopback TCP connections:\n" + "\n".join(external)


def test_app_is_configured_offline():
    app = stack_pids().get("app")
    if not app:
        pytest.skip("conversation app not running")
    try:
        env = dict(kv.split("=", 1) for kv in Path(f"/proc/{app[0]}/environ").read_bytes().decode().split("\0") if "=" in kv)
    except PermissionError:
        # Launched via `sg` (different group ID): /proc/<pid>/environ is unreadable. Fall back to the
        # app's startup log and the launcher that set the environment.
        log_path = ROOT / "local_backend/logs/conversation.log"
        # Make sure the log belongs to the running app, not an earlier run.
        started = float(subprocess.run(["ps", "-o", "etimes=", "-p", app[0]], capture_output=True,
                                       text=True, check=True).stdout)
        assert log_path.stat().st_mtime >= time.time() - started - 5, "conversation.log predates the running app"
        log = log_path.read_text()
        assert "connection mode: local" in log
        assert "Using direct Hugging Face realtime endpoint ws://127.0.0.1:8765" in log
        assert "Registered remote tool" not in log
        assert "export HF_HUB_OFFLINE=1" in (ROOT / "start_conversation.sh").read_text()
        return
    assert env.get("HF_REALTIME_CONNECTION_MODE") == "local"
    assert env.get("HF_REALTIME_WS_URL", "").startswith("ws://127.0.0.1:")
    assert env.get("HF_HUB_OFFLINE") == "1"


def test_gpu_placement_and_headroom():
    gpus = {}
    for line in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True).stdout.strip().splitlines():
        idx, uuid, used, total = [c.strip() for c in line.split(",")]
        gpus[uuid] = {"idx": idx, "used": int(used), "total": int(total)}
    procs = []
    for line in subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True).stdout.strip().splitlines():
        uuid, pid, used = [c.strip() for c in line.split(",")]
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace") if Path(f"/proc/{pid}").exists() else ""
        procs.append({"gpu": gpus[uuid]["idx"], "pid": pid, "mib": int(used), "cmd": cmd})
    record("gpu", gpus={g["idx"]: g["used"] for g in gpus.values()},
           procs=[{"gpu": p["gpu"], "mib": p["mib"], "what": "llm" if "llama-server" in p["cmd"] else
                   "speech" if "speech-to-speech" in p["cmd"] else "daemon" if "daemon" in p["cmd"] else "other"} for p in procs])
    for g in gpus.values():
        assert g["total"] - g["used"] >= GPU_MIN_HEADROOM_MIB, f"GPU {g['idx']} nearly full: {g}"
    llm_mib = sum(p["mib"] for p in procs if "llama-server" in p["cmd"])
    assert llm_mib > 15_000, f"LLM holds only {llm_mib} MiB of GPU memory; it may be partly on the CPU"
    speech_gpus = {p["gpu"] for p in procs if "speech-to-speech" in p["cmd"]}
    llm_gpus = {p["gpu"] for p in procs if "llama-server" in p["cmd"]}
    assert speech_gpus, "speech server holds no GPU memory"
    assert not speech_gpus & llm_gpus, f"speech {speech_gpus} shares a GPU with the LLM {llm_gpus}"
