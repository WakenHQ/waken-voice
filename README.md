# waken-voice

[![CI](https://github.com/WakenHQ/waken-voice/actions/workflows/ci.yml/badge.svg)](https://github.com/WakenHQ/waken-voice/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](https://github.com/WakenHQ/waken-voice/blob/main/LICENSE)

Voice [Source and Output](https://github.com/WakenHQ/waken) for
[Waken](https://github.com/WakenHQ/waken) — "nginx for AI agents." Unlike
`waken-claude`/`waken-gemini`/`waken-copilot`, this package doesn't wrap an
AI backend at all: it wraps a *channel*. `VoiceSource` turns a dropped
audio file into an `Event`; `MicrophoneSource` does the same for an
utterance spoken live after a wake word; `VoiceOutput` turns a `Response`
back into audio. All three talk to pluggable backends — OpenAI
(Whisper + TTS) by default, or swap in Groq (Whisper-backed transcription),
`gTTS` (free, no-API-key speech synthesis), and/or openWakeWord (free,
offline wake-word detection).

## Two ways in: file-drop or live microphone

`VoiceSource` watches a directory for **audio files appearing on disk**,
exactly like the built-in `FilesystemSource` watches for any file —
dropped there by a phone system, a Telegram/Slack voice-message webhook, a
push-to-talk recorder, or anything else. That's a small, honest, fully
testable scope that composes cleanly with whatever actually captures audio,
which is the same "small, composable pieces" philosophy Waken itself is
built on.

`MicrophoneSource` captures the microphone directly: it listens
continuously for a wake word (openWakeWord by default, fully offline), then
records the utterance that follows until it detects silence, transcribes
it with the same pluggable `Transcriber` `VoiceSource` uses, and dispatches
an `Event` — the wake-word-activated-assistant shape, without any cloud
wake-word service. Continuous capture and voice-activity segmentation over
real hardware is a genuinely hard, OS/driver-dependent problem, so treat
this the same way as `VoiceOutput`'s playback below: it's built to the same
design as the file-drop path and its state machine is fully unit-tested,
but the actual microphone/audio-stream wiring is **untested against real
hardware** by an agent with no microphone to speak into. See
[`MicrophoneSource`](#microphonesource) below.

## Install

```bash
pip install "waken-voice[openai]"   # default: Whisper + TTS via OpenAI
pip install "waken-voice[groq]"     # transcription via Groq instead
pip install "waken-voice[gtts]"     # synthesis via gTTS instead (free, no key)
pip install "waken-voice[mic]"      # live microphone input + wake word (MicrophoneSource)
pip install --no-deps openwakeword  # the wake-word engine itself — see below
```

Each provider is an extra, not a hard dependency — install only the ones
you use. `openai`, `groq`, `gtts`, and `mic` can all be combined in one
install (e.g. `pip install "waken-voice[groq,gtts,mic]"` for a fully
OpenAI-free setup).

- **OpenAI**: set `OPENAI_API_KEY` in the environment, or pass
  `api_key=...` to `OpenAIWhisperTranscriber`/`OpenAITTSSynthesizer`.
- **Groq**: set `GROQ_API_KEY`, or pass `api_key=...` to
  `GroqWhisperTranscriber`.
- **gTTS**: no API key — it calls Google Translate's public (undocumented)
  TTS endpoint directly.
- **mic**: no API key — `openwakeword` runs its pretrained ONNX models
  locally, and `sounddevice` talks to your OS's audio input (PortAudio)
  directly. Three things this needs beyond the extra above:
  - **PortAudio itself must be installed as a system library** —
    `sounddevice` wraps it via a compiled extension, so the pip package
    alone isn't enough. On Debian/Ubuntu: `sudo apt-get install
    libportaudio2`. On macOS: `brew install portaudio`. (Already present
    on most desktop Linux/macOS installs; bare containers and CI images
    are the case that needs this explicitly — see this repo's own
    `ci.yml`.)
  - **`openwakeword` itself needs `--no-deps`.** Upstream, it
    unconditionally depends on `tflite-runtime` on Linux, which has no
    published wheel for Python 3.12+ — a plain `pip install openwakeword`
    (or including it in the `mic` extra) fails to resolve at all on this
    package's supported Python versions. `waken-voice[mic]` installs
    everything `openwakeword` actually needs at runtime *except* that
    package; `OpenWakeWordDetector` defaults to
    `inference_framework="onnx"` (not openwakeword's own default of
    `"tflite"`) specifically so `tflite-runtime` is never required.
  - **Model files aren't bundled or auto-downloaded.** Run this once
    before first use:
    ```bash
    python -c "import openwakeword.utils; openwakeword.utils.download_models()"
    ```

## Usage

Default (OpenAI for both directions — no `transcriber`/`synthesizer`
arguments needed):

```python
from waken import Runtime
from waken_openai import OpenAIAdapter
from waken_voice import VoiceSource, VoiceOutput

runtime = Runtime()
runtime.target("openai", OpenAIAdapter())
runtime.source("voice-in", VoiceSource(watch="./voice-inbox", target="openai"))
runtime.output("voice-in", VoiceOutput(output_dir="./voice-outbox"))
runtime.run()
```

Groq for transcription, `gTTS` for synthesis:

```python
from waken import Runtime
from waken_openai import OpenAIAdapter
from waken_voice import VoiceSource, VoiceOutput, GroqWhisperTranscriber, GTTSSynthesizer

runtime = Runtime()
runtime.target("openai", OpenAIAdapter())
runtime.source(
    "voice-in",
    VoiceSource(
        watch="./voice-inbox",
        target="openai",
        transcriber=GroqWhisperTranscriber(),  # reads GROQ_API_KEY
    ),
)
runtime.output(
    "voice-in",
    VoiceOutput(output_dir="./voice-outbox", synthesizer=GTTSSynthesizer(lang="en")),
)
runtime.run()
```

Drop a `.wav`/`.mp3`/`.m4a`/`.ogg` file into `./voice-inbox`; `VoiceSource`
transcribes it and dispatches an `Event(payload={"prompt": <transcript>})`
to the `"openai"` target. `runtime.output(name, ...)` registers an `Output`
under a name resolved by `event.source` (or an explicit `event.output`) — so
registering `VoiceOutput` under `"voice-in"`, the same name as the source
above, means a `Response` to a voice-originated `Event` gets spoken back
by default; see [`docs/api-spec.md`
§3](https://github.com/WakenHQ/waken/blob/main/docs/api-spec.md#registration)
and [§9](https://github.com/WakenHQ/waken/blob/main/docs/api-spec.md#9-error-handling)
in the main `waken` repo for the full delivery-resolution rule.

Swap in `MicrophoneSource` for live wake-word listening instead of (or
alongside) `VoiceSource` — same `target`, same `Transcriber`, no file drop
required:

```python
from waken import Runtime
from waken_openai import OpenAIAdapter
from waken_voice import MicrophoneSource, VoiceOutput

runtime = Runtime()
runtime.target("openai", OpenAIAdapter())
runtime.source("mic-in", MicrophoneSource(target="openai"))
runtime.output("mic-in", VoiceOutput(output_dir="./voice-outbox"))
runtime.run()
```

Say the wake word ("hey jarvis" with the openWakeWord default model), then
speak; once you go quiet, the utterance is transcribed and dispatched the
same way a dropped file is.

## Providers

`VoiceSource`/`MicrophoneSource` take a `Transcriber` (anything with an
async `transcribe(path) -> str` method); `VoiceOutput` takes a
`Synthesizer` (anything with an async `synthesize(text) -> bytes` method);
`MicrophoneSource` additionally takes a `WakeWordDetector` (anything with a
sync `process(frame: bytes) -> str | None` method, given 16kHz mono 16-bit
PCM frames). All three are plain `Protocol`s, not base classes — write your
own by matching the method signature, same name-keyed-object philosophy as
`runtime.target("claude", ClaudeAdapter())` in core Waken.

| | Transcriber (STT) | Synthesizer (TTS) |
|---|---|---|
| **OpenAI** | `OpenAIWhisperTranscriber(model="whisper-1", **client_kwargs)` | `OpenAITTSSynthesizer(voice="alloy", model="tts-1", **client_kwargs)` |
| **Groq** | `GroqWhisperTranscriber(model="whisper-large-v3", **client_kwargs)` | — Groq has no TTS endpoint |
| **gTTS** | — Google Translate's TTS endpoint doesn't do STT | `GTTSSynthesizer(lang="en", **gtts_kwargs)` |

| | WakeWordDetector |
|---|---|
| **openWakeWord** | `OpenWakeWordDetector(wakeword_models=None, threshold=0.5, **model_kwargs)` |

`**client_kwargs` forwards to the provider's own async client
(`openai.AsyncOpenAI(...)` / `groq.AsyncGroq(...)`); `**gtts_kwargs` forwards
to `gtts.gTTS(...)` (e.g. `slow=True`); `**model_kwargs` forwards to
`openwakeword.model.Model(...)`. Each class imports its SDK lazily on
construction, so you only need the extra actually installed for whichever
class you use.

`gTTS` is worth knowing the shape of before relying on it: it's a thin
wrapper around Google Translate's public but undocumented TTS endpoint, not
an official API with an uptime or rate-limit contract, and it offers a
fixed accent-per-language voice rather than named-voice selection like
OpenAI's `alloy`/`nova`/... It's genuinely free and requires no signup,
which is the whole appeal — just don't reach for it if you need guaranteed
availability or voice control.

`OpenWakeWordDetector` ships pretrained models for a handful of wake words
(`"hey_jarvis"`, `"alexa"`, `"hey_mycroft"`, and a few others) and runs
them fully offline via ONNX Runtime — no key, no signup, same free-and-local
appeal as `gTTS` but for the opposite reason (it never leaves your
machine). Pass your own `.onnx`/`.tflite` model paths via `wakeword_models`
for a custom wake word.

## `VoiceSource`

```python
VoiceSource(
    watch,                  # directory to poll for new audio files
    target,                 # name of the registered Target to dispatch to
    interval=1.0,           # poll interval, seconds
    source_name="voice",
    transcriber=None,       # a Transcriber; defaults to OpenAIWhisperTranscriber()
)
```

Polls `watch` every `interval` seconds for new files, same mechanism as the
built-in `FilesystemSource` (see
[`waken/plugins/sources/filesystem.py`](https://github.com/WakenHQ/waken/blob/main/waken/plugins/sources/filesystem.py)) —
no OS-level file-watching dependency, files present before `start()` are the
baseline and never fire. Only `.wav`, `.mp3`, `.m4a`, and `.ogg` files are
transcribed; anything else is ignored silently, since handing a non-audio
file to a transcriber is a guaranteed, pointless error rather than a
useful attempt.

## `MicrophoneSource`

```python
MicrophoneSource(
    target,                        # name of the registered Target to dispatch to
    source_name="microphone",
    wake_word_detector=None,       # a WakeWordDetector; defaults to OpenWakeWordDetector()
    transcriber=None,              # a Transcriber; defaults to OpenAIWhisperTranscriber()
    sample_rate=16000,
    frame_samples=1280,            # ~80ms per frame at 16kHz, openWakeWord's expected chunk size
    silence_threshold=500.0,       # RMS amplitude below which a frame counts as silence
    silence_duration=1.0,          # seconds of trailing silence that ends an utterance
    max_utterance_duration=15.0,   # hard cap so a stuck-open mic can't record forever
    device=None,                   # sounddevice input device index/name; None = system default
    save_recordings=None,          # directory to keep each utterance's .wav; None = delete after transcribing
)
```

Opens a `sounddevice.InputStream` on `start()` and feeds every captured
frame through a small state machine: while **listening**, each frame goes
to `wake_word_detector`; a detection switches to **recording**, buffering
frames until they've been quiet for `silence_duration` seconds (or
`max_utterance_duration` is hit as a backstop), at which point the buffer
is written to a WAV file, handed to the same `Transcriber` `VoiceSource`
uses, and dispatched as an `Event(payload={"prompt": <transcript>})` —
then it's back to listening. A blank transcript (e.g. the transcriber
returning only whitespace) is dropped silently, same reasoning as
`VoiceOutput` skipping an empty `Response.text`.

**The audio-stream wiring (opening/reading `sounddevice.InputStream`) is
untested against real microphone hardware or OS/driver combinations** —
the same honest limitation as `VoiceOutput`'s playback below, for the same
reason: no agent building this autonomously has a microphone to speak
into. What *is* fully unit-tested is the state machine itself
(`_handle_frame`/`_finish_utterance`) driven with synthetic PCM frames, and
`start`/`stop`'s call shape against a mocked `sounddevice.InputStream`.
Treat this as a solid, testable state machine wrapped around one piece
you should verify yourself on your target hardware before relying on it.

## `VoiceOutput`

```python
VoiceOutput(
    output_dir,          # directory to write synthesized audio into
    synthesizer=None,    # a Synthesizer; defaults to OpenAITTSSynthesizer()
    play=True,           # attempt local playback after writing the file
)
```

Writes `{event.event_id}.mp3` under `output_dir`. A `Response` with no
`text` delivers nothing.

**Playback (`play=True`, the default) is best-effort and untested against
real audio hardware or OS combinations.** `_play` shells out to a platform
player (`afplay` on macOS, `paplay` on Linux, an unverified fallback
elsewhere) via `asyncio.create_subprocess_exec`, after the file is already
safely written to disk — a missing or failing player is logged, not raised.
This is the one piece of the package that genuinely can't be verified by an
agent with no speakers to listen to; treat it as a starting point for your
own environment, not a guarantee. Set `play=False` and hook your own
playback (or upload/streaming path) onto the written file if you need
something verified for your setup. (Same honest-about-limits spirit as
[`waken-copilot`'s
README](https://github.com/WakenHQ/waken-copilot/blob/main/README.md)
regarding its own unverified pieces.)

## Development

```bash
git clone https://github.com/WakenHQ/waken-voice
cd waken-voice
pip install -e ".[dev]"
pip install --no-deps openwakeword  # see the `mic` note under Install for why --no-deps
pytest
```

`dev` pulls in `openai`, `groq`, `gtts`, `sounddevice`, and openwakeword's
own runtime dependencies together so the full suite can run; `openwakeword`
itself needs the separate `--no-deps` install above. Tests mock each
provider's SDK client (`openai.AsyncOpenAI`, `groq.AsyncGroq`, `gtts.gTTS`,
`openwakeword.model.Model`, `sounddevice.InputStream`) and `VoiceOutput._play`
entirely — no real network access, API key, model download, or audio
hardware is required to run them.

## License

[MIT](https://github.com/WakenHQ/waken-voice/blob/main/LICENSE)
