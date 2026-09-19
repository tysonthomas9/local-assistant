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


_PROFILE_REQUESTS: dict = {}


def profile_request(profile: str) -> dict:
    """The system prompt and tools the app sends for `profile`, built from the current profile file.

    The app wraps the profile body in its own template (captured once in the fixture as
    system_prefix/system_suffix). Tool schemas come from the captured request where available
    (core tools, plus the app's system tools task_status/task_cancel), else from local_backend/tools.
    """
    if profile in _PROFILE_REQUESTS:
        return _PROFILE_REQUESTS[profile]
    text = (ROOT / f"local_backend/profiles/{profile}/profile.md").read_text()
    names = re.findall(r'"([a-z_]+)"', text.split("+++")[1].split("default_tools", 1)[1].split("]", 1)[0])
    captured = {t["function"]["name"]: t for t in APP_REQUEST["tools"]}
    missing = [n for n in names if n not in captured]
    code = (
        "import importlib.util, inspect, json, sys;"
        "from reachy_mini_conversation_app.tools.core_tools import Tool;"
        f"base = '{ROOT}/local_backend/tools/'; specs = {{}}"
        "\nfor name in sys.argv[1:]:"
        "\n    spec = importlib.util.spec_from_file_location(name, base + name + '.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"
        "\n    cls = next(c for c in vars(m).values() if inspect.isclass(c) and issubclass(c, Tool) and c is not Tool and c.name == name)"
        "\n    specs[name] = cls().spec()"
        "\nprint(json.dumps(specs))"
    )
    env = {**os.environ, "REACHY_REMINDERS_FILE": "/tmp/reachy_test_reminders.json"}  # never touch real reminders
    out = subprocess.run([str(APP_PYTHON), "-c", code, *missing], capture_output=True, text=True, check=True, env=env).stdout
    loaded = json.loads(out.strip().splitlines()[-1])
    to_chat = lambda sp: sp if "function" in sp else {"type": "function", "function": {k: v for k, v in sp.items() if k != "type"}}
    tools = [captured[n] if n in captured else to_chat(loaded[n]) for n in names]
    tools += [t for n, t in captured.items() if n not in names and n.startswith("task_")]  # added by the app itself
    system = APP_REQUEST["system_prefix"] + text.split("+++", 2)[2].strip() + APP_REQUEST["system_suffix"]
    _PROFILE_REQUESTS[profile] = {"system": system, "tools": tools, "names": names}
    return _PROFILE_REQUESTS[profile]


def chat(llm: OpenAI, user_content, history=None, profile="local_reachy", **kwargs):
    req = profile_request(profile)
    messages = [{"role": "system", "content": req["system"]}, *(history or []),
                {"role": "user", "content": user_content}]
    t0 = time.time()
    r = llm.chat.completions.create(model=MODEL, messages=messages, tools=req["tools"],
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
    ("Remind me in 10 minutes to check the oven.", "set_reminder"),
    ("Set a timer for 5 minutes.", "set_reminder"),
    ("Remind me at 6 pm to call my mom.", "set_reminder"),
    ("What reminders do I have?", "list_reminders"),
    ("Cancel my oven reminder.", ("cancel_reminder", "list_reminders")),  # listing first is also right
    ("Wake me up at 7 am.", "set_reminder"),
    ("Ring the alarm.", "play_sound"),
    ("Play a bell sound.", "play_sound"),
    ("Stop the alarm!", "stop_sound"),
    ("What's 17 times 23?", "calculate"),
    ("What's 15 percent of 240?", "calculate"),
    ("How many milliliters are in 3 and a half cups?", "convert_units"),
    ("Convert 72 Fahrenheit to Celsius.", "convert_units"),
    ("Add milk and eggs to my shopping list.", "lists"),
    ("What's on my to-do list?", "lists"),
    ("Talk like a noir detective from now on.", "switch_persona"),
    ("Can you be a Victorian butler?", "switch_persona"),
    ("Stop listening.", "listening"),
    ("Please don't listen to me for the next hour.", "listening"),
    ("Mute yourself for 10 minutes.", "listening"),
    ("Are you listening right now?", "listening"),
    ("Read me Alice in Wonderland.", "read_book"),
    ("Keep reading the book.", "read_book"),
    ("Stop reading, please.", "read_book"),
    ("What books do you have?", "read_book"),
]


SOUND_ARG_CASES = [
    ("Set a timer for 5 minutes.", "timer"),
    ("Set a 20 minute timer for the pasta.", "timer"),
    ("Wake me up at 7 am.", "alarm"),
    ("Set an alarm for 6:30 tomorrow morning.", "alarm"),
]


@pytest.mark.parametrize("prompt,sound", SOUND_ARG_CASES, ids=[f"{s}-{i}" for i, (_, s) in enumerate(SOUND_ARG_CASES)])
def test_llm_picks_reminder_sound(llm, prompt, sound):
    """Timers should ring the timer sound, alarms/wake-ups the alarm."""
    got = []
    for _ in range(TOOL_RUNS):
        msg, _ = chat(llm, prompt)
        calls = [tc for tc in (msg.tool_calls or []) if tc.function.name == "set_reminder"]
        got.append(json.loads(calls[0].function.arguments).get("sound") if calls else None)
    hits = sum(g == sound for g in got)
    record("reminder_sound", prompt=prompt, want=sound, got=got)
    assert hits >= TOOL_MIN_HITS, f"{prompt!r}: wanted sound={sound!r}, got {got}"


@pytest.mark.parametrize("prompt,tool", TOOL_CASES, ids=[f"{t if isinstance(t, str) else t[0]}-{i}" for i, (_, t) in enumerate(TOOL_CASES)])
def test_llm_calls_the_right_tool(llm, prompt, tool):
    results = []
    for _ in range(TOOL_RUNS):
        msg, dt = chat(llm, prompt)
        results.append(([tc.function.name for tc in (msg.tool_calls or [])], msg.content or "", round(dt, 2)))
    ok = (tool,) if isinstance(tool, str) else tool
    hits = sum(bool(set(ok) & set(names)) for names, _, _ in results)
    record("tool_call", tool=ok[0], prompt=prompt, hits=hits, runs=TOOL_RUNS, seconds=[r[2] for r in results])
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


def test_llm_announces_due_reminder(llm):
    # What reachy_scheduler injects through conversation.say when a reminder is due.
    replies = [chat(llm, "(Reminder due now) take the pizza out of the oven")[0] for _ in range(3)]
    for msg in replies:
        assert not msg.tool_calls, msg
        assert re.search(r"reminder", msg.content or "", re.I) and re.search(r"pizza|oven", msg.content or "", re.I), msg.content


def test_scheduler_parses_times_and_delivers(tmp_path, monkeypatch):
    """reachy_scheduler end to end against a fake /rpc: schedule, list, cancel, fire, overdue handling."""
    import importlib
    import sys
    from datetime import datetime, timedelta
    monkeypatch.setenv("REACHY_REMINDERS_FILE", str(tmp_path / "rem.json"))
    monkeypatch.setenv("REACHY_APP_RPC_URL", "ws://127.0.0.1:18779/rpc")
    monkeypatch.syspath_prepend(str(ROOT / "local_backend"))
    sys.modules.pop("reachy_scheduler", None)
    rs = importlib.import_module("reachy_scheduler")

    ref = datetime(2026, 9, 19, 14, 0).astimezone()
    assert rs.parse_due(None, "5:30 pm", ref=ref).strftime("%d %H:%M") == "19 17:30"
    assert rs.parse_due(None, "9am", ref=ref).strftime("%d %H:%M") == "20 09:00"   # already past -> tomorrow
    assert rs.parse_due(None, "noon", ref=ref).strftime("%d %H:%M") == "20 12:00"
    for bad in [(None, "25:00"), (None, "13 pm"), (5, "5pm"), (None, None), (-3, None), (None, "later")]:
        with pytest.raises(ValueError):
            rs.parse_due(*bad, ref=ref)

    said = []

    async def fake_rpc(ws):
        async for raw in ws:
            req = json.loads(raw)
            said.append(req["params"]["text"])
            await ws.send(json.dumps({"jsonrpc": "2.0", "method": "conversation.level", "params": {}}))  # noise
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": {"ok": True}}))

    async def scenario():
        async with websockets.serve(fake_rpc, "127.0.0.1", 18779):
            n = rs.now()
            rs.SCHEDULER._items = [
                {"id": "old001", "message": "water the plants", "due": (n - timedelta(hours=2)).isoformat(), "created": n.isoformat()},
                {"id": "late01", "message": "stretch", "due": (n - timedelta(minutes=5)).isoformat(), "created": n.isoformat()},
            ]
            rs.SCHEDULER.add("check the oven", n + timedelta(seconds=2))
            rs.SCHEDULER.add("call mom", n + timedelta(hours=3))
            assert [x["message"] for x in rs.SCHEDULER.cancel("mom")] == ["call mom"]
            await asyncio.sleep(6)

    asyncio.run(scenario())
    assert any(t.startswith("(Reminder due now, it was due at") and t.endswith("stretch") for t in said), said
    assert "(Reminder due now) check the oven" in said, said
    assert not any("plants" in t for t in said), "a 2-hour-old reminder should be dropped"
    assert not any("mom" in t for t in said), "a cancelled reminder fired"
    assert rs.SCHEDULER.pending() == []


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


def test_sounds_library_and_timing():
    """Generated library, play/stop, and that a reminder's ring blocks for the sound's real length."""
    code = (
        "import json, sys, time; sys.path.insert(0, sys.argv[1]);"
        "import reachy_sounds as snd, reachy_scheduler as rs;"
        "out = {'catalog': sorted(snd.catalog())};"
        "r = snd.PLAYER.play('alarm', repeat=0); time.sleep(1.5); out['ringing'] = snd.PLAYER.current; out['stopped'] = snd.PLAYER.stop();"
        "out['after_stop'] = snd.PLAYER.current; out['unknown'] = 'error' in snd.PLAYER.play('foghorn');"
        "t = time.monotonic(); rs.SCHEDULER._ring('timer'); out['timer_s'] = time.monotonic() - t;"
        "t = time.monotonic(); rs.SCHEDULER._ring('alarm'); out['alarm_s'] = time.monotonic() - t;"
        "t = time.monotonic(); rs.SCHEDULER._ring('none'); out['none_s'] = time.monotonic() - t;"
        "print(json.dumps(out))"
    )
    env = {**os.environ, "REACHY_SOUND_SINK": "fakesink sync=true",  # silent, but real-time like the speaker
           "REACHY_ALARM_SECONDS": "3", "REACHY_REMINDERS_FILE": "/tmp/reachy_test_reminders.json"}
    out = subprocess.run([str(APP_PYTHON), "-c", code, str(ROOT / "local_backend")], capture_output=True, text=True,
                         check=True, env=env).stdout
    r = json.loads(out.strip().splitlines()[-1])
    assert {"alarm", "timer", "chime", "bell", "beep", "success", "error", "wake_up"} <= set(r["catalog"]), r["catalog"]
    assert r["ringing"] == "alarm" and r["stopped"] == "alarm" and r["after_stop"] is None, r
    assert r["unknown"], "unknown sound names should return an error"
    assert 2.5 < r["timer_s"] < 5, f"timer ring should last ~2.85 s, took {r['timer_s']:.2f}"
    assert 3 <= r["alarm_s"] < 6.5, f"alarm should ring ~3 s (+ at most one ~2 s cycle), took {r['alarm_s']:.2f}"
    assert r["none_s"] < 0.2


def _run_tools(code: str, **env):
    """Run `code` (which prints one JSON line) under the app venv with local_backend on sys.path."""
    full = "import asyncio, importlib.util, json, sys\nsys.path.insert(0, sys.argv[1])\n" + \
           "def load(n, c):\n    spec = importlib.util.spec_from_file_location(n, sys.argv[1] + '/tools/' + n + '.py')\n" + \
           "    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m, getattr(m, c)()\n" + code
    out = subprocess.run([str(APP_PYTHON), "-c", full, str(ROOT / "local_backend")], capture_output=True, text=True,
                         check=True, env={**os.environ, **env}).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_calculator_and_unit_conversion():
    r = _run_tools("""
_, C = load('calculate', 'Calculate'); _, U = load('convert_units', 'ConvertUnits')
exprs = ['17*23', '17 x 23', '15% of 80', '2^10', 'sqrt(2)*10', '(3+4)*5/2', '100 divided by 7', 'sin(30)', '12 plus 5 minus 3',
         '5 squared', 'min(1,2)', 'max(3, 7)', 'round(3.14159, 2)', '1,234,567 * 2',
         '1/0', '2**100000', "__import__('os')", '(1).__class__', "open('x')", 'a+b', '9'*400, '('*500 + '1' + ')'*500,
         'round(1, -10**9)', 'floor(exp(709))', 'min(5)']
convs = [(72, '°F', 'celsius'), (3.5, 'cups', 'ml'), (10, 'km', 'miles'), (1, 'lb', 'grams'), (2, 'tablespoons', 'teaspoons'),
         (60, 'mph', 'km/h'), (3, 'kg', 'celsius'), (1, 'parsec', 'km')]
async def main():
    return [[await C(None, expression=e) for e in exprs], [await U(None, value=v, from_unit=a, to_unit=b) for v, a, b in convs]]
print(json.dumps(asyncio.run(main())))
""")
    calc, conv = r
    results = [c.get("result") for c in calc]
    assert results[:14] == [391, 391, 12, 1024, pytest.approx(14.1421356), 17.5, pytest.approx(14.2857142), pytest.approx(0.5), 14, 25,
                            1, 7, 3.14, 2469134]
    # 1/0, huge power, code injection, names, huge inputs, round() digit bomb, huge function result, 1-arg min
    assert all("error" in c for c in calc[14:]), calc[14:]
    assert [round(c["result"], 2) if "result" in c else "error" for c in conv] == [22.22, 828.06, 6.21, 453.59, 6.0, 96.56, "error", "error"]


def test_lists_roundtrip(tmp_path):
    f = str(tmp_path / "lists.json")
    first = _run_tools("""
_, L = load('lists', 'Lists')
async def main():
    return [await L(None, action='add', list_name='Shopping List', items=['milk', 'eggs', 'Bread']),
            await L(None, action='add', list_name='groceries', items=['MILK', 'coffee']),
            await L(None, action='add', list_name='to do', item='call the plumber'),
            await L(None, action='remove', list_name='shopping', item='egg')]
print(json.dumps(asyncio.run(main())))
""", REACHY_LISTS_FILE=f)
    assert first[1]["already_on_list"] == ["MILK"] and first[2]["list"] == "todo" and first[3]["removed"] == ["eggs"]
    second = _run_tools("""
_, L = load('lists', 'Lists')
async def main():
    return [await L(None, action='read', list_name='shopping'), await L(None, action='list_lists'),
            await L(None, action='clear', list_name='todo'), await L(None, action='read', list_name='todo')]
print(json.dumps(asyncio.run(main())))
""", REACHY_LISTS_FILE=f)   # a new process: persisted?
    assert second[0]["items"] == ["milk", "Bread", "coffee"]
    assert second[1]["lists"] == {"shopping": 3, "todo": 1}
    assert second[2]["cleared"] == 1 and second[3]["items"] == []


@pytest.mark.online
def test_currency_conversion():
    r = _run_tools("""
_, X = load('convert_currency', 'ConvertCurrency')
async def main():
    return [await X(None, amount=50, from_currency='euros', to_currency='dollars'),
            await X(None, amount=5, from_currency='EUR', to_currency='zorkmids')]
print(json.dumps(asyncio.run(main())))
""")
    assert r[0]["from"] == "EUR" and r[0]["to"] == "USD" and 30 < r[0]["result"] < 90, r[0]
    assert "error" in r[1]


def test_personas_generated_and_up_to_date():
    import importlib.util as iu
    spec = iu.spec_from_file_location("make_personas", ROOT / "local_backend/make_personas.py")
    mp = iu.module_from_spec(spec); spec.loader.exec_module(mp)
    expected = mp.generate(write=False)
    assert len(expected) == 2 * len(mp.personas()) >= 20
    for path, content in expected.items():
        assert path.exists() and path.read_text() == content, f"{path.parent.name} is stale: run local_backend/make_personas.py"
    for suffix, base in mp.BASES.items():
        base_tools = mp.tool_names(mp.split_profile((ROOT / f"local_backend/profiles/{base}/profile.md").read_text())[0])
        assert "switch_persona" in base_tools
        for name in mp.personas():
            front, body = mp.split_profile((ROOT / f"local_backend/profiles/local_{name}{suffix}/profile.md").read_text())
            assert mp.tool_names(front) == base_tools, name
            assert "## TOOL & MOVEMENT RULES" in body and "## SPEECH RULES" in body, name


def test_persona_matching():
    r = _run_tools("""
m, _ = load('switch_persona', 'SwitchPersona')
qs = ['be a Victorian butler', 'the noir detective please', 'talk like David Attenborough', 'be a chef', 'go back to normal',
      'Mars rover', 'mad scientist', 'time traveller', 'a pirate']
print(json.dumps([m.match(q) for q in qs]))
""")
    assert r == ["victorian_butler", "noir_detective", "nature_documentarian", "cosmic_kitchen", "", "mars_rover",
                 "mad_scientist_assistant", "time_traveler", None]


def test_privacy_mute_and_timer():
    r = _run_tools("""
import types, time
import reachy_bridge as b; b.install()
import reachy_listening as rl
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.moves import BreathingMove, MovementState
m, T = load('listening', 'Listening')
class FakeMM:
    def __init__(self): self.state = MovementState(); self.move_queue = []; self.cleared = 0
    def queue_move(self, mv): self.state.current_move = mv
    def clear_move_queue(self): self.cleared += 1; self.state.current_move = None
mm = FakeMM(); deps = types.SimpleNamespace(movement_manager=mm); rl.CUE_CHECK_S = 0.2
async def main():
    s = LocalStream(types.SimpleNamespace(output_queue=asyncio.Queue()), types.SimpleNamespace(media=types.SimpleNamespace(audio=None)))
    out = {'before': await T(deps, action='status'), 'stop': await T(deps, action='stop', minutes=0.03)}
    await asyncio.sleep(0.5)
    cur = mm.state.current_move; h, ants, _ = cur.evaluate(5.0)
    out.update(muted=s._mic_muted, pose=type(cur).__name__, preemptable=isinstance(cur, BreathingMove), antennas=list(ants))
    await asyncio.sleep(2)
    out.update(after_timer=s._mic_muted, pose_cleared=mm.cleared >= 1)
    out['default'] = await T(deps, action='stop'); out['resume'] = await T(deps, action='resume'); out['resumed'] = s._mic_muted
    out['bad'] = await T(deps, action='stop', minutes=-5)
    return out
print(json.dumps(asyncio.run(main())))
""")
    assert r["before"]["listening"] is True and r["stop"]["muted"] is True
    assert r["muted"] is True and r["pose"] == "MutedPose" and r["preemptable"] and r["antennas"] == [-2.2, 2.2]
    assert r["after_timer"] is False and r["pose_cleared"], "timer should unmute and drop the pose"
    assert r["default"]["minutes_left"] == 60 and r["resume"]["listening"] is True and r["resumed"] is False
    assert "error" in r["bad"]


def test_llm_mute_duration(llm):
    got = []
    for _ in range(TOOL_RUNS):
        msg, _ = chat(llm, "Stop listening for 10 minutes.")
        calls = [tc for tc in (msg.tool_calls or []) if tc.function.name == "listening"]
        got.append(json.loads(calls[0].function.arguments) if calls else None)
    ok = sum(bool(g) and g.get("action") == "stop" and float(g.get("minutes") or 0) == 10 for g in got)
    record("mute_minutes", hits=ok, runs=TOOL_RUNS, got=got)
    assert ok >= TOOL_MIN_HITS, got


def _reader_module(monkeypatch, **env):
    import importlib
    import sys
    import types
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.syspath_prepend(str(ROOT / "local_backend"))
    monkeypatch.setitem(sys.modules, "reachy_bridge", types.ModuleType("reachy_bridge"))
    sys.modules.pop("reachy_reader", None)
    return importlib.import_module("reachy_reader")


def test_book_parsing(monkeypatch):
    """Committed excerpt: Gutenberg header/footer, a short table of contents, heading-like sentences."""
    rr = _reader_module(monkeypatch)
    b = rr.parse(FIXTURES / "books" / "alice_excerpt.txt")
    assert b.title == "Alice's Adventures in Wonderland"
    assert [t for t, _ in b.chapters] == ["Chapter 1. Down the Rabbit-Hole", "Chapter 11. Who Stole the Tarts?"]
    first = b.chunks[b.chapters[0][1]]
    assert first.startswith("Chapter 1. Down the Rabbit-Hole. Alice was beginning")
    body = " ".join(b.chunks[b.chapters[0][1]:])
    assert "Licence text" not in body and "[Illustration" not in body and body.rstrip().endswith("THE END")
    assert "Part of the reason was that nobody had told her." in body, "an ordinary sentence must not become a chapter"
    assert max(len(c.split()) for c in b.chunks) <= 180
    assert rr._sentences('He said "Oh dear!" Then he left.') == ['He said "Oh dear!"', "Then he left."]
    number = lambda h: rr._heading_number(rr._HEADING.match(h).group(2)) if rr._HEADING.match(h) else None
    assert [number(h) for h in ("CHAPTER XI. Who Stole the Tarts?", "Chapter 12", "BOOK ONE", "Chapter the First",
                                "Chapter Mix", "Chapter IIII", "Part of the reason")] == [11, 12, 1, 1, None, None, None]
    # number words followed by an ordinary sentence aren't headings; compound number words are
    assert [number(h) for h in ("Book one was better than book two.", "Part one of the plan was simple.",
                                "Letter one arrived on Monday", "Chapter 3 of the manual says so.",
                                "Chapter Twenty-One", "CHAPTER THIRTY-TWO. The End", "Chapter Twenty-Ten")] == \
        [None, None, None, None, 21, 32, None]


def test_book_find_and_bookmarks(tmp_path, monkeypatch):
    books = tmp_path / "books"
    books.mkdir()
    (books / "alice.txt").write_text((FIXTURES / "books" / "alice_excerpt.txt").read_text())
    rr = _reader_module(monkeypatch, REACHY_BOOKS_DIR=str(books), REACHY_BOOKMARKS_FILE=str(tmp_path / "marks.json"))
    assert rr.find("the wonderland book").name == "alice.txt" and rr.find("moby dick") is None
    assert rr.bookmark("alice") == -1
    rr.save_mark("alice", 4)
    assert rr.bookmark("alice") == 4 and rr.last_book() == "alice"


def test_reader_pauses_when_a_passage_never_arrives(tmp_path, monkeypatch):
    """No transcript within GEN_TIMEOUT_S (e.g. backend restarted) must pause, not skip the passage."""
    import importlib
    import sys
    import types
    monkeypatch.setenv("REACHY_BOOKMARKS_FILE", str(tmp_path / "marks.json"))
    monkeypatch.syspath_prepend(str(ROOT / "local_backend"))
    for m in ("reachy_bridge", "reachy_reader"):
        sys.modules.pop(m, None)
    bridge = importlib.import_module("reachy_bridge")
    rr = importlib.import_module("reachy_reader")
    rr.GEN_TIMEOUT_S, rr.GAP_S = 0.5, 0.0
    sent = []

    async def scenario():
        class Handler:
            async def _safe_response_create(self, **kw):
                sent.append(kw)   # swallowed: no transcript ever comes back
        bridge._stream = types.SimpleNamespace(handler=Handler(), _asyncio_loop=asyncio.get_running_loop(), _mic_muted=False)
        book = rr.parse(FIXTURES / "books" / "alice_excerpt.txt")
        rr.READER.start(book, 2)
        await asyncio.sleep(0.3)
        bridge.publish("assistant_transcript_done")   # the tool's confirmation finished
        for _ in range(100):                          # timeout 0.5 s after the send; allow a loaded machine
            if not rr.READER.reading:
                break
            await asyncio.sleep(0.1)
        return book

    book = asyncio.run(scenario())
    assert len(sent) == 1, "only the first passage should have been sent"
    assert rr.READER.reading is False and rr.bookmark(book.key) == 2, "bookmark must stay on the unread passage"


def test_reader_waits_for_the_conversation_and_pauses_on_wake_word(tmp_path, monkeypatch):
    """Never queue a passage behind an active reply; "Hey Jarvis" in the pause stops the next passage;
    stop_now only cancels a passage that is really active on the server (else it would cut the model's reply)."""
    import importlib
    import sys
    import types
    monkeypatch.setenv("REACHY_BOOKMARKS_FILE", str(tmp_path / "marks.json"))
    monkeypatch.syspath_prepend(str(ROOT / "local_backend"))
    for m in ("reachy_bridge", "reachy_reader"):
        sys.modules.pop(m, None)
    bridge = importlib.import_module("reachy_bridge")
    rr = importlib.import_module("reachy_reader")
    rr.GEN_TIMEOUT_S, rr.GAP_S, rr.LEAD_S = 5, 1.0, 0.0
    sent, cancels = [], []

    async def until(cond, timeout=5.0):
        for _ in range(int(timeout / 0.05)):
            if cond():
                return True
            await asyncio.sleep(0.05)
        return False

    async def scenario():
        async def cancel():
            cancels.append(1)

        class Handler:
            def __init__(self):
                self._response_done_event = asyncio.Event()   # clear: the model is still answering
                self._pending_responses = asyncio.Queue()
                self.connection = types.SimpleNamespace(response=types.SimpleNamespace(cancel=cancel))

            async def _safe_response_create(self, **kw):
                sent.append(kw)
        h = Handler()
        bridge._stream = types.SimpleNamespace(handler=h, _asyncio_loop=asyncio.get_running_loop(), _mic_muted=False,
                                               clear_audio_queue=lambda: None)
        book = rr.parse(FIXTURES / "books" / "alice_excerpt.txt")
        out = {}
        rr.READER.start(book, 0)
        await asyncio.sleep(0.1)
        bridge.publish("assistant_transcript_done")    # the tool's confirmation finished...
        await asyncio.sleep(1.0)
        out["sent_while_busy"] = len(sent)             # ...but its response is still active
        h._response_done_event.set()
        out["sent_when_idle"] = await until(lambda: len(sent) == 1)
        bridge.publish("response_created")
        out["inflight"] = await until(lambda: rr.READER._inflight)
        bridge.publish("assistant_transcript_done")    # passage 0 read; the pause begins
        await asyncio.sleep(0.3)
        bridge.publish("wake_word")                    # "Hey Jarvis" in the pause
        out["paused"] = await until(lambda: not rr.READER.reading)
        await asyncio.sleep(1.2)                       # past the end of the pause
        out.update(sent_after_wake=len(sent), mark=rr.bookmark(book.key))

        for label, created in (("cancels_not_inflight", False), ("cancels_inflight", True)):
            rr.READER.start(book, 1)
            await asyncio.sleep(0.1)
            bridge.publish("assistant_transcript_done")
            n = len(sent)
            await until(lambda: len(sent) > n)
            if created:
                bridge.publish("response_created")
                await until(lambda: rr.READER._inflight)
            await rr.READER.stop_now()
            out[label] = len(cancels)
        return out

    r = asyncio.run(scenario())
    assert r["sent_while_busy"] == 0, "a passage must not be queued behind an active reply"
    assert r["sent_when_idle"] and r["inflight"]
    assert r["paused"] and r["sent_after_wake"] == 1, "wake word in the pause must stop the next passage"
    assert r["mark"] == 1, "passage 0 was heard in full: resume with the next one"
    assert r["cancels_not_inflight"] == 0, "no passage active on the server: don't cancel the model's reply"
    assert r["cancels_inflight"] == 1


@pytest.mark.online
def test_gutenberg_download(tmp_path, monkeypatch):
    import importlib
    import sys
    import types
    monkeypatch.setenv("REACHY_BOOKS_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(ROOT / "local_backend"))
    monkeypatch.setitem(sys.modules, "reachy_bridge", types.ModuleType("reachy_bridge"))
    sys.modules.pop("reachy_reader", None)
    rr = importlib.import_module("reachy_reader")
    path = rr.gutenberg_download("the time machine wells")
    book = rr.parse(path)
    assert "time machine" in book.title.lower() and len(book.chunks) > 50


def test_llm_bedtime_story_is_made_up(llm):
    """'Tell me a bedtime story' (no book named) should be told by the model, not start the book reader."""
    calls = []
    for _ in range(TOOL_RUNS):
        msg, _ = chat(llm, "Tell me a bedtime story.")
        calls.append([tc.function.name for tc in (msg.tool_calls or [])])
    ok = sum("read_book" not in c for c in calls)
    record("bedtime_story", hits=ok, runs=TOOL_RUNS, calls=calls)
    assert ok >= TOOL_MIN_HITS, calls


def test_wake_word_gate(tmp_path):
    """openWakeWord + gate: speech without the wake word never leaves; 'Hey Jarvis' opens a window;
    mute/hard-mute and book reading behave as documented (reachy_wake.py)."""
    piper = ROOT / ".venv" / "bin" / "python"
    clips = {"wake": "Hey Jarvis, what time is it?", "other": "Hey Reachy, tell me a joke about robots and the weather."}
    code = ("import sys, wave; from piper import PiperVoice; v = PiperVoice.load(sys.argv[1])\n"
            "for name, text in [a.split('=', 1) for a in sys.argv[3:]]:\n"
            "    w = wave.open(sys.argv[2] + '/' + name + '.wav', 'wb'); v.synthesize_wav(text, w); w.close()")
    subprocess.run([str(piper), "-c", code, str(ROOT / "voices/en_US-lessac-medium.onnx"), str(tmp_path),
                    *[f"{k}={v}" for k, v in clips.items()]], check=True, capture_output=True)
    r = _run_tools(r"""
import time, types, wave, numpy as np
from scipy.signal import resample_poly
import reachy_bridge as b; b.install()
import reachy_wake as rw, reachy_listening as rl, reachy_reader
from reachy_mini_conversation_app.console import LocalStream
d = sys.argv[2]
def frames(name):
    with wave.open(d + '/' + name + '.wav') as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768; sr = w.getframerate()
    a = np.concatenate([np.zeros(8000), resample_poly(a, 16000, sr), np.zeros(16000)]).astype(np.float32)
    st = np.stack([a, a], axis=1); return [st[i:i+512] for i in range(0, len(st) - 511, 512)]
gate = rw.Gate(rw.Detector('hey_jarvis', 0.5), window_s=1.0, followup_s=1.0); b.subscribe(gate.on_activity)
s = LocalStream(types.SimpleNamespace(output_queue=asyncio.Queue()), types.SimpleNamespace(media=types.SimpleNamespace(audio=None)))
def run(name):
    return sum(len(x) for f in frames(name) for x in gate.process(f, 16000, s._mic_muted, rl.is_hard_muted())) / 16000
out = {'other': run('other')}
out['wake'] = run('wake'); out['open_after_wake'] = gate.is_open(); time.sleep(1.2); out['closed_later'] = not gate.is_open()
gate.extend(1.0); b.publish('response_created'); time.sleep(0.6); b.publish('assistant_audio_delta'); time.sleep(0.6)
out['kept_open_while_answering'] = gate.is_open(); time.sleep(1.2)
rl.mute(10, hard=False); out['muted_wake_forwarded'] = run('wake'); out['unmuted_by_wake'] = not s._mic_muted
time.sleep(1.2); rl.mute(10, hard=True); out['hard_forwarded'] = run('wake'); out['still_hard_muted'] = s._mic_muted; rl.resume()
time.sleep(1.2); reachy_reader.READER.reading = True; gate.extend(0.3); time.sleep(0.1)
for _ in range(8): b.publish('assistant_audio_delta'); time.sleep(0.1)
out['reading_holds_open'] = gate.is_open()
print(json.dumps(out))
""".replace("sys.argv[2]", repr(str(tmp_path))))
    assert r["other"] == 0.0, "speech without the wake word must not be forwarded"
    assert r["wake"] > 2.5 and r["open_after_wake"] and r["closed_later"], "wake word + pre-roll should be forwarded"
    assert r["kept_open_while_answering"]
    assert r["muted_wake_forwarded"] > 0.5 and r["unmuted_by_wake"]
    assert r["hard_forwarded"] == 0.0 and r["still_hard_muted"]
    assert r["reading_holds_open"] is False


def test_wake_phrase_hey_reachy(tmp_path):
    """sherpa-onnx keyword spotter with the phrase as text: "Hey Reachy ..." opens the gate (pre-roll included);
    near misses and ordinary speech mentioning "reaching" don't. Routing between the two engines."""
    import pytest as _pytest
    kws = ROOT / "local_backend" / "models" / "kws" / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
    if not (kws / "tokens.txt").exists():
        _pytest.skip("keyword model missing: run local_backend/cache_models.sh")
    piper = ROOT / ".venv" / "bin" / "python"
    clips = {"wake": "Hey Reachy, what time is it?", "short": "Hey Reachy.",
             "reggie": "Hey Reggie, how are you?", "reaching": "I'm reaching out to him. Teach me something. That's peachy."}
    code = ("import sys, wave; from piper import PiperVoice; v = PiperVoice.load(sys.argv[1])\n"
            "for name, text in [a.split('=', 1) for a in sys.argv[3:]]:\n"
            "    w = wave.open(sys.argv[2] + '/' + name + '.wav', 'wb'); v.synthesize_wav(text, w); w.close()")
    subprocess.run([str(piper), "-c", code, str(ROOT / "voices/en_US-lessac-medium.onnx"), str(tmp_path),
                    *[f"{k}={v}" for k, v in clips.items()]], check=True, capture_output=True)
    r = _run_tools(r"""
import time, types, wave, numpy as np
from scipy.signal import resample_poly
import reachy_bridge as b; b.install()
import reachy_wake as rw
d = sys.argv[2]
def frames(name):
    with wave.open(d + '/' + name + '.wav') as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768; sr = w.getframerate()
    a = np.concatenate([np.zeros(8000), resample_poly(a, 16000, sr), np.zeros(16000)]).astype(np.float32)
    st = np.stack([a, a], axis=1); return [st[i:i+512] for i in range(0, len(st) - 511, 512)]
out = {'routing': [rw.uses_openwakeword(w) for w in ('hey reachy', 'hey_jarvis', 'alexa', '/x/custom.onnx', 'ok robot')]}
det = rw.make_detector('hey reachy')
out['engine'] = type(det).__name__
try:
    rw.PhraseDetector('hey reachy ééé')
    out['bad_phrase'] = 'accepted'
except ValueError:
    out['bad_phrase'] = 'rejected'
for name in ('reggie', 'reaching', 'wake', 'short'):
    gate = rw.Gate(rw.make_detector('hey reachy'), window_s=1.0, followup_s=1.0)
    out[name] = sum(len(x) for f in frames(name) for x in gate.process(f, 16000, False, False)) / 16000
print(json.dumps(out))
""".replace("sys.argv[2]", repr(str(tmp_path))))
    assert r["routing"] == [False, True, True, True, False]
    assert r["engine"] == "PhraseDetector" and r["bad_phrase"] == "rejected"
    assert r["reggie"] == 0.0 and r["reaching"] == 0.0, "near misses must not open the gate"
    assert r["wake"] > 1.5 and r["short"] > 0.5, "'Hey Reachy' (+ pre-roll) should be forwarded"


def test_lists_removal_priority_and_names(tmp_path):
    r = _run_tools("""
import reachy_lists as rl
rl.add('shopping', ['vegan eggs', 'eggs', 'eggplant', 'milk'])
out = [rl.list_key('my shopping list'), rl.list_key('The to do list'), rl.list_key('my list'),
       rl.remove('shopping', 'egg')[1], rl.remove('shopping', 'vegan')[1], rl.remove('shopping', 'plant')[1], rl.read('shopping')[1]]
rl.add('shopping', ['tomato']); out.append(rl.remove('shopping', 'tomatoes')[1])
open(rl.STATE_FILE, 'w').write('{broken')
out.append(rl.lists())
print(json.dumps(out))
""", REACHY_LISTS_FILE=str(tmp_path / "lists.json"))
    assert r[:3] == ["shopping", "todo", "notes"]
    assert r[3:7] == [["eggs"], ["vegan eggs"], ["eggplant"], ["milk"]]
    assert r[7] == ["tomato"], "plural 'tomatoes' removes 'tomato'"
    assert r[8] == {} and any(p.name.startswith("lists.corrupt") for p in tmp_path.iterdir()), "corrupt file backed up"


def test_mute_state_follows_the_ui_toggle():
    r = _run_tools("""
import types
import reachy_bridge as b; b.install()
import reachy_listening as rl
from reachy_mini_conversation_app.console import LocalStream
s = LocalStream(types.SimpleNamespace(output_queue=asyncio.Queue()), types.SimpleNamespace(media=types.SimpleNamespace(audio=None)))
rl.mute(30, hard=True)
out = {'hard_before': rl.is_hard_muted()}
s._mic_muted = False              # the web UI un-mutes
out['after_ui_unmute'] = rl.status()
s._mic_muted = True               # ...and mutes again: a plain mute, not our old hard one
out['hard_after_ui_mute'] = rl.is_hard_muted(); out['status_ui_mute'] = rl.status()
print(json.dumps(out))
""")
    assert r["hard_before"] is True
    assert r["after_ui_unmute"] == {"listening": True, "muted": False}
    assert r["hard_after_ui_mute"] is False and "resumes_at" not in r["status_ui_mute"]


def test_mute_survives_the_per_frame_sync():
    """With --wake, every mic frame calls is_hard_muted() -> _sync_with_mic(). A frame landing between mute()
    recording its state and flipping the mic used to look like a UI un-mute and drop the hard flag and timer.
    Force a frame into exactly that moment: it must wait for mute() to finish."""
    r = _run_tools("""
import threading, time, types
import reachy_bridge as b; b.install()
import reachy_listening as rl
from reachy_mini_conversation_app.console import LocalStream
s = LocalStream(types.SimpleNamespace(output_queue=asyncio.Queue()), types.SimpleNamespace(media=types.SimpleNamespace(audio=None)))
real_set = b.set_mic_muted
def set_mic_muted(muted):
    if muted:   # a mic frame arrives right before the flag flips
        t = threading.Thread(target=rl.is_hard_muted); t.start(); t.join(0.2)
    return real_set(muted)
b.set_mic_muted = set_mic_muted
rl.mute(30, hard=True)
time.sleep(0.3)
print(json.dumps(rl.status()))
""")
    assert r["muted"] is True and r["hard"] is True and r["minutes_left"] == 30, r


def test_bridge_playback_clock_survives_odd_frames():
    """The push_audio_sample wrapper must never raise (it runs inside the app's play loop) and count samples right."""
    r = _run_tools("""
import types, numpy as np
from reachy_mini.media import media_manager
media_manager.MediaManager.push_audio_sample = lambda self, data: None   # stand-in for the real pipeline push
import reachy_bridge as b; b.install()                                  # wraps the stand-in
fake = types.SimpleNamespace(get_output_audio_samplerate=lambda: 16000)
out = {}
for name, arr in [('mono_1s', np.zeros(16000, np.float32)), ('n_by_ch_0.5s', np.zeros((8000, 2), np.float32)),
                  ('ch_by_n_0.5s', np.zeros((2, 8000), np.float32)), ('zero_d', np.float32(0.0))]:
    b._play_end = 0.0
    try:
        media_manager.MediaManager.push_audio_sample(fake, arr)
        out[name] = round(b.audio_seconds_left(), 1)
    except Exception as e:
        out[name] = 'raised ' + type(e).__name__
print(json.dumps(out))
""")
    assert r == {"mono_1s": 1.0, "n_by_ch_0.5s": 0.5, "ch_by_n_0.5s": 0.5, "zero_d": 0.0}, r


LIST_ARG_CASES = [
    ("Add milk to my shopping list.", {"action": "add", "list": "shopping", "item": "milk"}),
    ("Put call the dentist on my to-do list.", {"action": "add", "list": "todo", "item": "dentist"}),
    ("Remove eggs from the shopping list.", {"action": "remove", "list": "shopping", "item": "egg"}),
    ("What's on my shopping list?", {"action": "read", "list": "shopping"}),
]


@pytest.mark.parametrize("prompt,want", LIST_ARG_CASES, ids=[w["action"] + str(i) for i, (_, w) in enumerate(LIST_ARG_CASES)])
def test_llm_lists_arguments(llm, prompt, want):
    got, ok = [], 0
    for _ in range(TOOL_RUNS):
        msg, _ = chat(llm, prompt)
        calls = [tc for tc in (msg.tool_calls or []) if tc.function.name == "lists"]
        a = json.loads(calls[0].function.arguments) if calls else {}
        got.append(a)
        name = " ".join(str(a.get("list_name", "")).lower().replace("-", " ").split())
        name = {"to do": "todo", "to do list": "todo", "shopping list": "shopping", "todos": "todo"}.get(name, name)
        items = " ".join(map(str, a.get("items") or [])) + " " + str(a.get("item") or "")
        ok += (a.get("action") == want["action"] and name == want["list"]
               and ("item" not in want or want["item"] in items.lower()))
    record("lists_args", prompt=prompt, hits=ok, runs=TOOL_RUNS)
    assert ok >= TOOL_MIN_HITS, got


@pytest.mark.parametrize("prompt,tool", [("What time is it?", "get_time"), ("Do a little dance.", "dance"),
                                         ("Go back to being yourself.", "switch_persona"), ("What's 12 times 12?", "calculate")])
def test_persona_keeps_tools(llm, prompt, tool):
    """A generated persona must still use its tools (not just role-play)."""
    hits = 0
    for _ in range(TOOL_RUNS):
        msg, _ = chat(llm, prompt, profile="local_victorian_butler")
        hits += tool in [tc.function.name for tc in (msg.tool_calls or [])]
    record("persona_tool", tool=tool, prompt=prompt, hits=hits, runs=TOOL_RUNS)
    assert hits >= TOOL_MIN_HITS, f"{tool}: {hits}/{TOOL_RUNS}"


# --- Online tools (local_reachy_web profile: --web) ----------------------------------------

WEB_TOOLS = ("get_weather", "web_search", "tech_news", "play_radio", "stop_radio")


def web_profile_body() -> str:
    text = (ROOT / "local_backend/profiles/local_reachy_web/profile.md").read_text()
    return text.split("+++", 2)[2]


def test_web_profile_adds_exactly_the_online_tools():
    offline = (ROOT / "local_backend/profiles/local_reachy/profile.md").read_text()
    web = (ROOT / "local_backend/profiles/local_reachy_web/profile.md").read_text()
    for tool in WEB_TOOLS:
        assert f'"{tool}"' in web and f'"{tool}"' not in offline
    assert "pollen_robotics_" not in web
    assert "entirely offline" not in web


@pytest.fixture(scope="session")
def web_request():
    return profile_request("local_reachy_web")


WEB_TOOL_CASES = [
    ("What's the weather like in Tokyo right now?", "get_weather"),
    ("Is it going to rain in Seattle today?", "get_weather"),
    ("Can you look up who won the last Formula One race?", "web_search"),
    ("Search the web for the price of a Raspberry Pi 5.", "web_search"),
    ("Any tech news today?", "tech_news"),
    ("What's on Hacker News right now?", "tech_news"),
    ("Play some jazz.", "play_radio"),
    ("Put on BBC Radio 1.", "play_radio"),
    ("Stop the music, please.", "stop_radio"),
    ("Set a timer for 3 minutes.", "set_reminder"),
    ("How much is 50 euros in dollars?", "convert_currency"),
]


@pytest.mark.parametrize("prompt,tool", WEB_TOOL_CASES, ids=[f"{t}-{i}" for i, (_, t) in enumerate(WEB_TOOL_CASES)])
def test_web_llm_calls_the_right_tool(llm, web_request, prompt, tool):
    hits = 0
    for _ in range(TOOL_RUNS):
        r = llm.chat.completions.create(model=MODEL, max_tokens=300, tools=web_request["tools"], tool_choice="auto",
                                        messages=[{"role": "system", "content": web_request["system"]},
                                                  {"role": "user", "content": prompt}],
                                        extra_body={"reasoning_effort": "none"})
        hits += tool in [tc.function.name for tc in (r.choices[0].message.tool_calls or [])]
    record("web_tool_call", tool=tool, prompt=prompt, hits=hits, runs=TOOL_RUNS)
    assert hits >= TOOL_MIN_HITS, f"{tool}: {hits}/{TOOL_RUNS} for {prompt!r}"


def test_web_search_pagination_with_fake_searxng():
    """Uneven, overlapping SearXNG pages -> clean 5-result pages, has_more, cache, 'No more results'."""
    code = r"""
import asyncio, importlib.util, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
PAGES = {1: [f"https://site{i}.com/a" for i in range(7)],
         2: ["http://www.site3.com/a/", "https://site7.com/a", "https://site8.com/a", "https://site9.com/a"],
         3: [f"https://site{i}.com/a" for i in range(10, 16)], 4: []}
hits = []
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        n = int(parse_qs(urlparse(self.path).query)["pageno"][0]); hits.append(n)
        body = json.dumps({"results": [{"title": u, "content": "c", "url": u} for u in PAGES.get(n, [])]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ["REACHY_SEARXNG_URL"] = f"http://127.0.0.1:{srv.server_address[1]}"
spec = importlib.util.spec_from_file_location("web_search", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
W = m.WebSearch()
async def main():
    out = []
    for q, pg in (("Test  Query", 1), ("test query", 2), ("test query", 3), ("test query", 4), ("test query", 5), ("TEST query", 1)):
        before = len(hits); r = await W(None, query=q, page=pg)
        out.append({"page": pg, "sites": [x["source"] for x in r["results"]], "has_more": r["has_more"],
                    "note": r.get("note"), "requests": hits[before:]})
    return out
print(json.dumps(asyncio.run(main())))
"""
    out = subprocess.run([str(APP_PYTHON), "-c", code, str(ROOT / "local_backend/tools/web_search.py")],
                         capture_output=True, text=True, check=True).stdout
    p1, p2, p3, p4, p5, again = json.loads(out.strip().splitlines()[-1])
    site = lambda u: re.sub(r"^https?://(www\.)?", "", u).split(".")[0]
    assert [site(u) for u in p1["sites"]] == [f"site{i}" for i in range(5)] and p1["has_more"] and p1["requests"] == [1]
    assert [site(u) for u in p2["sites"]] == [f"site{i}" for i in range(5, 10)], "page 2 must continue page 1, without the duplicate site3"
    assert p3["requests"] == [], "page 3 should come from the cache"
    assert [site(u) for u in p4["sites"]] == ["site15"] and p4["has_more"] is False
    assert p5["sites"] == [] and p5["note"] == "No more results."
    assert again["requests"] == [] and [site(u) for u in again["sites"]] == [f"site{i}" for i in range(5)], "same query, other spelling -> cache"
    all_sites = p1["sites"] + p2["sites"] + p3["sites"] + p4["sites"]
    assert len({site(u) for u in all_sites}) == len(all_sites), "duplicates across pages"


@pytest.mark.online
def test_web_search_pages_real():
    code = r"""
import asyncio, importlib.util, json, sys
spec = importlib.util.spec_from_file_location("web_search", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
W = m.WebSearch()
async def main():
    return [await W(None, query="Reachy Mini robot", page=p) for p in (1, 2)]
print(json.dumps(asyncio.run(main())))
"""
    out = subprocess.run([str(APP_PYTHON), "-c", code, str(ROOT / "local_backend/tools/web_search.py")],
                         capture_output=True, text=True, check=True).stdout
    p1, p2 = json.loads(out.strip().splitlines()[-1])
    assert len(p1["results"]) == 5 and p1["has_more"], p1
    assert p2["results"] and not ({x["source"] for x in p1["results"]} & {x["source"] for x in p2["results"]}), "pages overlap"


def test_web_llm_asks_for_next_page(llm, web_request):
    """After a search, 'show me more' should call web_search again with the same query and page 2."""
    first = {"query": "best robot vacuum 2026", "page": 1, "has_more": True, "results": [
        {"title": f"Robot vacuum review {i}", "snippet": "…", "source": f"https://example{i}.com/review"} for i in range(5)]}
    history = [
        {"role": "user", "content": "Search the web for the best robot vacuum this year."},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "id": "call_ws1", "function": {
            "name": "web_search", "arguments": json.dumps({"query": "best robot vacuum 2026"})}}]},
        {"role": "tool", "tool_call_id": "call_ws1", "content": json.dumps(first)},
        {"role": "assistant", "content": "Reviewers mostly recommend a handful of mid-range models this year."},
    ]
    got = []
    for _ in range(TOOL_RUNS):
        r = llm.chat.completions.create(model=MODEL, max_tokens=300, tools=web_request["tools"], tool_choice="auto",
                                        messages=[{"role": "system", "content": web_request["system"]}, *history,
                                                  {"role": "user", "content": "Can you show me more results?"}],
                                        extra_body={"reasoning_effort": "none"})
        calls = [tc for tc in (r.choices[0].message.tool_calls or []) if tc.function.name == "web_search"]
        got.append(json.loads(calls[0].function.arguments) if calls else None)
    ok = sum(bool(g) and int(g.get("page") or 1) == 2 and "vacuum" in g.get("query", "").lower() for g in got)
    record("web_next_page", hits=ok, runs=TOOL_RUNS, got=got)
    assert ok >= TOOL_MIN_HITS, got


@pytest.mark.online
def test_web_tools_return_data():
    """Calls the three tools for real (internet + local SearXNG)."""
    code = (
        "import asyncio, importlib.util, json;"
        f"base = '{ROOT}/local_backend/tools/';"
        "\ndef load(n, c):"
        "\n    spec = importlib.util.spec_from_file_location(n, base + n + '.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return getattr(m, c)()"
        "\nasync def main():"
        "\n    return [await load('get_weather','GetWeather')(None, place='Paris'),"
        "\n            await load('web_search','WebSearch')(None, query='Reachy Mini robot'),"
        "\n            await load('tech_news','TechNews')(None, count=2)]"
        "\nprint(json.dumps(asyncio.run(main())))"
    )
    out = subprocess.run([str(APP_PYTHON), "-c", code], capture_output=True, text=True, check=True).stdout
    weather, search, news = json.loads(out.strip().splitlines()[-1])
    assert "Paris" in weather.get("place", ""), weather
    assert re.search(r"-?\d+°[FC]", weather.get("now", "")), weather
    assert search.get("results"), search
    assert len(news.get("headlines", [])) >= 2, news


@pytest.mark.online
def test_radio_plays_and_stops():
    """Finds a jazz station and plays it into a fakesink (no sound), then stops it."""
    code = (
        "import asyncio, importlib.util, json;"
        f"base = '{ROOT}/local_backend/tools/';"
        "\ndef load(n, c):"
        "\n    spec = importlib.util.spec_from_file_location(n, base + n + '.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return getattr(m, c)()"
        "\nasync def main():"
        "\n    play, stop = load('play_radio','PlayRadio'), load('stop_radio','StopRadio')"
        "\n    a = await play(None, query='jazz'); await asyncio.sleep(2); b = await stop(None); c = await stop(None)"
        "\n    d = await play(None, query='zzqxnonexistentstation')"
        "\n    return [a, b, c, d]"
        "\nprint(json.dumps(asyncio.run(main())))"
    )
    env = {**os.environ, "REACHY_RADIO_SINK": "fakesink"}
    out = subprocess.run([str(APP_PYTHON), "-c", code], capture_output=True, text=True, check=True, env=env).stdout
    played, stopped, again, nonsense = json.loads(out.strip().splitlines()[-1])
    assert played.get("playing"), played
    assert stopped.get("stopped") == played["playing"], stopped
    assert "note" in again and "error" in nonsense


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
        req = profile_request("local_reachy")
        tools = [{"type": "function", **t["function"]} for t in req["tools"]]
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime",
                      "instructions": req["system"], "tools": tools, "audio": {"output": {"voice": "Aiden"}}}}))

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


class _FakeApp:
    """Stands in for the conversation app so the book reader can be tested against the real speech server.

    handler._safe_response_create sends response.create over a Realtime websocket, one at a time
    (like the app's sender loop); audio deltas feed the bridge's playback clock at SPEEDUP x real time;
    transcript.done publishes "assistant_transcript_done" through the bridge, as the app does.
    """
    SPEEDUP = 2.0   # must stay below the speech server's generation speed (~3.5x real time) so audio can pile up

    def __init__(self, loop, ws):
        self._asyncio_loop, self.ws = loop, ws
        self.queue: asyncio.Queue = asyncio.Queue()
        self.transcripts: list[str] = []
        self.max_buffered = 0.0
        self.handler = self
        self._mic_muted = False

    async def _safe_response_create(self, **kwargs):
        await self.queue.put(kwargs)

    async def run(self, bridge, rate: int):
        while True:
            kwargs = await self.queue.get()
            await self.ws.send(json.dumps({"type": "response.create", **kwargs}))
            text = ""
            while True:
                ev = json.loads(await self.ws.recv())
                if ev["type"].endswith("audio.delta"):
                    bridge.add_played_audio(len(base64.b64decode(ev["delta"])) / 2 / rate / self.SPEEDUP)
                    self.max_buffered = max(self.max_buffered, bridge.audio_seconds_left())
                elif ev["type"].endswith("audio_transcript.delta"):
                    text += ev["delta"]
                elif ev["type"].endswith("audio_transcript.done"):
                    self.transcripts.append(text)
                    bridge.publish("assistant_transcript_done")
                elif ev["type"] in ("response.done", "error"):
                    break


@pytest.mark.realtime
def test_book_reader_reads_verbatim_paces_and_pauses(tmp_path, monkeypatch):
    import importlib
    import sys
    book_src = FIXTURES / "books" / "alice_excerpt.txt"
    monkeypatch.setenv("REACHY_BOOKMARKS_FILE", str(tmp_path / "books.json"))
    monkeypatch.syspath_prepend(str(ROOT / "local_backend"))
    for m in ("reachy_bridge", "reachy_reader"):
        sys.modules.pop(m, None)
    bridge = importlib.import_module("reachy_bridge")
    rr = importlib.import_module("reachy_reader")
    rr.LEAD_S, rr.GAP_S = 0.0, 1.5 / _FakeApp.SPEEDUP   # same pacing, compressed time
    book = rr.parse(book_src)
    start = book.chapters[0][1]

    async def scenario():
        first, ws = None, None
        for _ in range(15):   # the previous test's session may take a few seconds to release the slot
            try:
                ws = await websockets.connect(REALTIME, max_size=None)
                first = json.loads(await ws.recv())
                if first.get("type") == "session.created":
                    break
                await ws.close()
                await asyncio.sleep(1)
            except OSError as e:
                pytest.fail(f"speech server not reachable: {e}")
            except websockets.exceptions.ConnectionClosed:
                await asyncio.sleep(1)
        if not first or first.get("type") != "session.created":
            pytest.skip("speech server slot in use: stop the conversation app to run realtime tests")
        rate = (first["session"].get("audio", {}).get("output", {}).get("format", {}) or {}).get("rate", 24000)
        app = _FakeApp(asyncio.get_running_loop(), ws)
        bridge._stream = app
        pump = asyncio.create_task(app.run(bridge, rate))
        bridge.publish("assistant_transcript_done")  # the tool's confirmation "finished"
        rr.READER.start(book, start)
        await asyncio.sleep(0.5)
        bridge.publish("assistant_transcript_done")
        # wait until 2 passages are transcribed and a 3rd is in flight, then interrupt it
        t0 = time.time()
        while len(app.transcripts) < 2 and time.time() - t0 < 120:
            await asyncio.sleep(0.2)
        bridge.publish("interrupted")
        await asyncio.sleep(1.0)
        playing = rr.READER.pos   # the passage that was cut off (or the next one, if it came in the pause)
        paused = {"reading": rr.READER.reading, "mark": rr.bookmark(book.key)}
        pump.cancel()
        await ws.close()
        return app, playing, paused

    app, playing, paused = asyncio.run(scenario())
    norm = lambda t: re.sub(r"[^a-z0-9' ]", " ", t.lower().replace("’", "'").replace("-", " ")).split()
    for i, text in enumerate(app.transcripts[:2]):
        from difflib import SequenceMatcher
        ref, hyp = norm(book.chunks[start + i]), norm(text)
        ratio = SequenceMatcher(None, ref, hyp).ratio()
        diffs = [(op, " ".join(ref[a1:a2]), " ".join(hyp[b1:b2])) for op, a1, a2, b1, b2
                 in SequenceMatcher(None, ref, hyp).get_opcodes() if op != "equal"]
        record("reader_passage", passage=start + i, match_ratio=round(ratio, 3), diffs=diffs[:10])
        assert ratio >= 0.95, f"passage {start + i} not read verbatim ({ratio:.3f}): {text[:120]!r}"
    record("reader_pacing", max_buffered_s=round(app.max_buffered * _FakeApp.SPEEDUP, 1))
    one_passage = max(len(c.split()) for c in book.chunks[start:start + 3]) / 2.0  # >= ~2 words/s of speech
    assert app.max_buffered * _FakeApp.SPEEDUP <= one_passage + 2, "reader buffered more than one passage"
    assert paused["reading"] is False, "reader should pause on interruption"
    assert paused["mark"] == playing, "bookmark should point at the passage that was playing"
    assert playing >= start + 1


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
