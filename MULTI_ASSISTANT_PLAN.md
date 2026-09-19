# Plan: several assistants on one robot, picked by wake word

Drafted and vetted by Fable on 2026-09-19. Status: **plan approved in part (decisions below); not started.** Paths are relative to `~/codebase/robots`. "App" = `reachy_mini_conversation_app/src/reachy_mini_conversation_app/`, "S2S" = `third_party/speech-to-speech/src/speech_to_speech/`. Upstream stays unmodified: everything is wrapped or patched from `local_backend/`.

## Goal

"Hey Jarvis, …" talks to Jarvis; "Hey Marvin, …" talks to Marvin. Each assistant has its own personality, voice, conversation history, memory and data, and neither sees the other's. Follow-ups without a wake word go to whoever answered last.

## Decisions (user, 2026-09-19)

1. **Everything per assistant.** One data folder per assistant, `local_backend/state/assistants/<name>/`, holding history, memory, lists, reminders, book bookmarks and its current style (persona text and voice). Chosen as easier in the long run: one rule ("use the active assistant's folder") instead of a shared-or-separate call per feature, and new tools inherit it. Stateless things (radio, sound effects, the book files) stay shared. Reminders keep an owner. When one comes due while another assistant is active, the active one announces it as "Jarvis's reminder: …" (no session switch).
2. **Jarvis** = the calm butler, voice Ryan. **Marvin** = the gloomy robot from Hitchhiker's Guide, voice Eric.
3. **History persists across app restarts:** the last ~20 turns per assistant, cleared by voice ("forget our conversation").
4. **`switch_persona` stays, scoped to the current assistant:** it restyles that assistant (personality text and/or voice). The style is saved in the assistant's folder, so it persists; the wake word and data folder don't change. "Go back to normal" restores the assistant's base style.

## Facts (vetted by Fable against the code)

- **Which word fired:** openWakeWord's `Model.predict` returns a score per loaded model; our `Detector.feed` currently collapses them with `max()` (`local_backend/reachy_wake.py:105`). Per-model thresholds are supported.
- **Live session changes:** S2S deep-merges `session.update`, and the next generation uses the new instructions, tools and voice (`runtime_config.py:10-24,78-81`; `LLM/language_model.py:632-637`; the TTS reads the voice per response).
- **No way to clear a session's history:** there is no `conversation.item.delete`, and `truncate` is a logged no-op (`websocket_router.py:464-470`). History lives in the per-connection state (`service.py:334-357`). So isolating histories needs a new session, and there is no cheaper server-side reset.
- **Replay works:** `conversation.item.create` accepts user and assistant messages and passes them to the LLM as prior turns (`LLM/chat.py:1200-1227, 298-304, 728+`). Caveats:
  - items sent during an active response are deferred (`handlers/conversation.py:128-143`);
  - ids must start with `msg_`/`call_` (`chat.py:69`);
  - beyond 30 user turns (`chat_size`) the server compacts the history with an extra Ollama call, so replays stay under 30 turns.
- **Memory:** patching `memory.memory_path_for_instance` covers remember, forget and prompt injection (`memory.py:148,157,175,194`).
- **Transcripts:** the transcript observer carries only user and assistant text (App `huggingface_realtime.py:829,838`). Tool calls need their own hook (wrap `LocalStream._dispatch_transcript` and the tool manager, as `reachy_bridge` wraps `_dispatch_activity`).
- **Rejected alternatives:**
  - *every turn out-of-band:* server VAD turns always create in-band responses, and tool calls in out-of-band responses aren't recorded, so tool results would fail;
  - *instructions-only switching:* instant, but each assistant sees the other's turns;
  - *a warm spare pipeline:* it doesn't keep history, and the fix below makes it unnecessary;
  - *one live session per assistant (B):* feasible with patches, but it duplicates the handler lifecycle, the idle policy and the wiring, and doubles STT/TTS GPU memory for no gain once switches are fast.

## Design: one session, swapped per assistant

When the *other* assistant's wake word fires (also inside an open follow-up window):

1. **Interrupt** the current assistant if it's speaking (flush, cancel) and stop the book reader.
2. **Hold** mic audio in the gate: the pre-roll and everything after it. The app drops frames while there's no connection (App `huggingface_realtime.py:957`), so holding is required. Cap: 10 s.
3. **Switch:**
   - set the incoming assistant's profile, voice and data folder;
   - close the session, **wait for the server to free its slot** (poll `GET /v1/pool` until idle), then trigger the rebuild.
   - Today the app reconnects 7–11 ms after closing, while the slot takes ~40–50 ms to drain. The server rejects the connection and the app retries after 1–1.5 s. Measured in speech.log: every past persona switch hit this.
4. **Replay** the incoming assistant's saved turns (under 30) in place of the greeting. The hook is `_send_startup_greeting_prompt`, which runs once the connection exists and no response is active. Router-built handlers are marked "greeting sent" via a wrapped `_handler_factory`.
5. **Release** the held audio after the replay (not merely after connecting). Burst forwarding is fine: the server splits it into 512-sample chunks, and VAD/STT run faster than real time.

Expected switch time after the fix: roughly the reconnect plus the replay (well under 1 s), to be measured.

## Build order

0. **Measure:**
   - switch time (RPC `personalities.apply` → "Realtime session updated"), idle and mid-response;
   - slot drain while a generation is in flight;
   - how often `session_limit_reached` occurs.
1. **Reconnect fix** (poll the pool before rebuilding). This also speeds up today's `switch_persona`.
2. **Multi-word detector with arbitration.** On the first frame over a threshold, wait up to 2 more frames (160 ms) and pick the highest score, with a margin ≥ 0.2. Per-model thresholds (Marvin stricter, e.g. 0.5, since it fired at 0.4 on "Hey Jarvis, stop") and per-model cooldowns. The gate publishes `wake_word:<name>`. Offline tests: Piper "Hey Jarvis" and "Hey Marvin" clips through both models.
3. **Registry and personas:**
   - `local_backend/assistants.json`: name, wake word, threshold, base profile, base voice, data folder, and a default assistant at boot;
   - `make_personas.py` builds `local_jarvis[_web]` and `local_marvin[_web]`, each profile naming only its own wake word.
4. **Router** `local_backend/reachy_assistants.py`. It keeps its state outside tool files, because tool files are re-executed on each profile reload. It covers:
   - hold → switch → replay → release;
   - the wrapped `_handler_factory` (greeting suppression);
   - the voice: set `LocalStream._voice_override` per assistant. Today a persisted UI voice beats every profile, so both assistants would sound the same;
   - the boot profile: set the default assistant before the first connect, because `startup_settings.json` otherwise overrides it.
5. **Per-assistant state:**
   - capture history, including tool calls as short text notes;
   - memory path patch;
   - lists, reminders (with owner) and bookmarks resolved from the active assistant's folder;
   - saved style for `switch_persona`.
6. **Interactions:**
   - **the web UI persona picker:** wrap `stream.apply_personality`, so it restyles the active assistant (decision 4) instead of bypassing the router and re-greeting;
   - **`switch_persona`:** uses the same path;
   - **reminders:** `say` fails while disconnected, and the scheduler retries 3 × 5 s, so a switch under 15 s is fine. Queue a reminder that lands mid-replay until the replay is done;
   - **mute:** a wake word while muted un-mutes and routes;
   - **idle dance:** cancelled by the restart, which is fine;
   - **follow-ups:** they stay with the active assistant.
7. **Realtime isolation test:** tell Jarvis a fact, switch to Marvin (he must not know it), switch back (Jarvis must). Plus a live check on the robot.
8. **Docs:** a LOCAL_CONVERSATION.md section and the launcher flag (`--assistants`, implies `--wake`).

## Remaining decisions

5. **Default assistant at boot:** Jarvis, or whoever was active last?
6. **The web UI's persona picker:** restyle the active assistant (proposed, same as `switch_persona`), or switch assistants?
7. **Switch latency:** is well under 1 s (expected after the fix) acceptable?
