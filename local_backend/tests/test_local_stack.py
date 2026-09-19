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
