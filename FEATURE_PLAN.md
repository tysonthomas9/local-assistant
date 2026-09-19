# Plan: calculator, lists, privacy mute, wake word, persona switcher, storyteller

Planned 2026-09-19 by a Fable planning agent (read-only, grounded in the code), spot-checked by Claude. Status: **all six built, tested offline and reviewed twice by Fable (2026-09-19); live checks of the muted pose, persona voices and sound levels pending** — details in LOCAL_CONVERSATION.md. Paths are relative to `~/codebase/robots`. "App" = `reachy_mini_conversation_app/src/reachy_mini_conversation_app/`, "S2S" = `third_party/speech-to-speech/src/speech_to_speech/`.

**Verified before writing this doc:**
- Mute drops mic frames before they're sent (App `console.py:881`: `if audio_frame is not None and not self._mic_muted`).
- `personalities.apply` / `voices.apply` exist as RPC methods (App `personality_routes.py:374-392`).
- S2S supports out-of-band responses that skip the pending-tool check (`handlers/response.py:735-753`, `is_out_of_band`).
- `openwakeword` 0.6.0 is on PyPI.

## 0. Foundation: in-process bridge (S)

Tools only get `reachy_mini`, `movement_manager` and `instance_path` (App `tools/core_tools.py:36-46`), but mute, wake word, storyteller and persona switching need the live conversation stream.

`local_backend/reachy_bridge.py` will be installed by `run_app.py`, like the existing UI-host patch. It will:
- keep a reference to the `LocalStream`;
- publish its activity events (`user_speech_started`, `response_created`, `assistant_transcript_done`, …) and barge-ins (`clear_audio_queue`);
- expose `_mic_muted`, the handler (`say`, `_safe_response_create`), the event loop and `apply_personality`.

The RPC (`ws://127.0.0.1:7860/rpc`) stays for cross-process callers such as the scheduler and the tests.

## 1. Calculator and unit conversion (S)

- **`calculate`** (both profiles): a safe `ast`-whitelist evaluator.
  - Allowed: numbers, `+ - * / // % **`, parentheses, `sqrt sin cos tan log ln exp abs round floor ceil min max`, `pi e`, and "15% of 80".
  - Caps on exponent size and result digits. No `eval`.
  - Returns `result` and a `spoken` form.
- **`convert_units`** (both): a hand-written table (length, mass, volume incl. cups/tbsp/tsp/fl oz, temperature, area, speed, data, time) with aliases ("°F", "fahrenheit", "cups").
- **`convert_currency`** (web only): frankfurter.app (ECB rates, no key), cached for a day.
- **Profile rule:** never do arithmetic or conversions without the tool.
- **Tests:** a table of expressions, including rejected inputs (names, attributes, `__import__`, huge powers); tool choice for "17 times 23", "millilitres in 3½ cups", "72 °F to Celsius", "50 euros in dollars".

## 2. Lists (S/M)

- **One action tool, `lists`** (both profiles): `action=add|remove|read|clear|list_lists`, `list_name` (shopping, todo, notes, …), `item`; removal matches words.
- **Storage:** state in `local_backend/reachy_lists.py` → `local_backend/state/lists.json` (git-ignored).
- **Kept separate from `remember`/`forget`:** memory holds ≤60 stable facts injected into every prompt; lists are bigger and not in the prompt.
- **Profile rule:** shopping, to-do and notes go in lists; ask before clearing.
- **Tests:** a round trip including persistence; tool choice and arguments for add, read, remove and clear.

## 3. Privacy mute (M)

- **One action tool, `listening`** (both): `action=stop|resume|status`, `minutes` (default 60; 0 = until resumed). It sets `_mic_muted` through the bridge.
- **What mute does:** no audio leaves the process. Reminders and radio still play ("deaf, not silent").
- **Un-muting:** a timer, the web UI's mic toggle, or the local wake word (the detector runs on the dropped frames). `hard=true` disables the detector too.
- **Visual cue:** a held "antennas down" move while muted; its direction has to be checked live.
- **Tests:** stop/resume/timer against a fake `/rpc`; tool choice for "stop listening", "don't listen for an hour", "you can listen again", "are you listening?". Live: no turns in `speech.log` while muted.

## 4. Wake word (L)

**Recommended: a gate in the app, before audio is sent.**
- **Detection:** the bridge wraps the handler's `receive` (App `huggingface_realtime.py:947`). Each frame goes to a ~1.5 s ring buffer and an **openWakeWord** detector (Apache-2.0, offline, onnxruntime).
- **On "hey …":** forward the buffer, so the first words of the command survive, and open a listening window of ~8 s. The window stays open while the robot speaks, and ~6 s after, so follow-ups and interruptions don't need the wake word again.
- **Reminders** (`conversation.say`) open the window too, so "stop the alarm" works.
- **Privacy:** gated audio never leaves the process.
- **Opt-in:** a `--wake` launcher flag.

**Update (2026-09-19):** "Hey Reachy" now works without training, via sherpa-onnx's open-vocabulary keyword spotter (LOCAL_CONVERSATION.md, "Wake phrase"). **Original plan:** start with a **pre-trained word** (e.g. "hey jarvis"; the exact list is unverified). Train a custom **"Hey Reachy"** model later with openWakeWord's notebook, estimated at ~1 h on this GPU (unverified).

**Rejected:**
- **Porcupine:** it needs an online AccessKey.
- **Transcript gating** (drop turns whose text lacks "Reachy"): Parakeet mishears the name, every TV sentence still costs speech-to-text and LLM time, and cancelling is racy. It could still be an optional cheap extra.

**Tests:** the detector on recorded clips plus a new wake-word clip; the gate's state machine with synthetic events. Live: false triggers with the TV on, and barge-in during a long reply.

## 5. Persona switcher (M)

**How switching works now:** `personalities.apply` → `LocalStream.apply_personality` (App `console.py:449-462`). It reloads the tools, restarts the backend session (**conversation history is lost**) and plays the new persona's greeting.

**Only our profiles directory is listed**, and names can't clash with the built-ins. So:
- **Generator `local_backend/make_personas.py`:** turns the 14 upstream personas (mars_rover, noir_detective, victorian_butler, …) into `local_<name>` and `local_<name>_web`, each with our tool list and rules plus the persona text and a voice. The output is committed, and a test checks it's up to date.
- **Tool `switch_persona`:** matches a persona name loosely ("normal"/"yourself" = the base profile) and keeps the current `_web`/offline mode. It **returns immediately**, then switches in the background after the confirmation has been spoken. Awaiting the switch inside the tool would drop the confirmation.

**Voices:** Aiden, Ryan, Dylan, Eric, Ono_Anna, Serena, Sohee, Uncle_Fu and Vivian are in the app's list (`config.py:51-61`). Which ones our Qwen3-TTS GGUF build supports is unverified; probe them first. A voice picked in the UI overrides the profile's voice.

**Tests:** generator idempotence; name matching; tool choice for "switch to the pirate", "be a Victorian butler", "go back to normal". Live: the greeting and voice, and tools still working after a switch.

## 6. Storyteller / book reader (L)

**How to speak long text:**
- **Rejected: `conversation.say`.** It paraphrases, and each chunk becomes a fake user turn in the history.
- **Recommended: out-of-band `response.create`.** The app sends it via `handler._safe_response_create(response={conversation:"none", input:[passage], instructions:"read exactly…", …})`.
  - It skips the pending-tool block, stays out of the history, and an interruption cancels it (S2S `handlers/audio.py:170-172`).
  - Same voice as the robot; nearly word for word (to be measured).
- **Fallback, verbatim mode:** Piper TTS pushed into the app's audio queue. Truly word for word, but a different voice.
- **Not an option:** a TTS-only endpoint, because S2S has none.

**Design:**
- **`local_backend/reachy_reader.py`:**
  - a library of `local_backend/books/*.txt` (git-ignored, with a README);
  - chapter detection and ~150-word chunks;
  - bookmarks in `state/books.json`;
  - a reader thread that sends the next chunk when the last one finishes speaking, and pauses on interruption.
- **Tool `read_book`:** `action=start|continue|pause|stop|next_chapter|list|status|download`. `download` fetches from Project Gutenberg, web profile only.
- **Bedtime stories** (made up, not from a book): a profile rule to tell it in parts and ask whether to continue.

**Biggest unknown:** does the robot hear itself reading? With `~/.asoundrc`, the SDK doesn't add its software echo cancellation, so it would have to come from the USB audio board. The radio shares this risk.
- **First live test:** a 2-minute read while watching `speech.log` for spurious turns.

**Tests:** the chunker, bookmarks and chapters offline; a Realtime test sending one out-of-band chunk and comparing the transcript to the passage; tool choice for "read me a book", "next chapter", "stop reading", "tell me a bedtime story".

## Cross-cutting

- **Tool count:** 25 in the web profile today → ~34. Keep descriptions short, prefer action tools, and re-run the whole tool-choice suite after each addition. `dance`, at 7/8, is the canary.
- **Priority:** mute > wake gate > everything else.
- **Interactions:**
  - Reminders speak while muted, and open the wake gate.
  - A persona switch resets the session: stop reading first. Lists and reminders are files, so they survive.
  - Radio and reading share the echo-cancellation unknown.
- **Testable offline:** calculator, lists, the persona generator, the reader logic, the gate state machine, tool choice.
- **Needs the robot:** the antenna cue, echo cancellation, wake-word false triggers, persona greeting and voice.

## Build order

1. Bridge (S)
2. calculate / convert (S) ∥ lists (S/M) ∥ persona generator (S)
3. Privacy mute (M) ∥ switch_persona (M)
4. Storyteller (L)
5. Wake word (L); it shares the gate code with mute

## Decisions (recommended defaults)

1. Merge into action tools (`lists`, `listening`, `read_book`): **yes**
2. Wake word: openWakeWord with a pre-trained word now, train "Hey Reachy" later; opt-in `--wake`: **yes**
3. Un-mute by timer + UI + local wake word, with a hard mode: **yes**
4. Storyteller voice: out-of-band LLM reading, Piper as the verbatim fallback, after measuring accuracy: **yes**
5. Generate all 14 personas × offline/web and commit them: **yes**
6. Currency via frankfurter.app (web only): **yes**
7. Books in `local_backend/books/` (git-ignored except a README), Gutenberg download with `--web`: **yes**
