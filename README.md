# waken-voice

[![CI](https://github.com/WakenHQ/waken-voice/actions/workflows/ci.yml/badge.svg)](https://github.com/WakenHQ/waken-voice/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](https://github.com/WakenHQ/waken-voice/blob/main/LICENSE)

Voice [Source and Output](https://github.com/WakenHQ/waken) for
[Waken](https://github.com/WakenHQ/waken) — "nginx for AI agents." Unlike
`waken-claude`/`waken-gemini`/`waken-copilot`, this package doesn't wrap an
AI backend at all: it wraps a *channel*. `VoiceSource` turns an audio file
into an `Event`; `VoiceOutput` turns a `Response` back into audio. Both are
backed by OpenAI's audio API (Whisper for speech-to-text, TTS for
text-to-speech).

## What this is NOT: a live microphone listener

Continuous microphone capture with voice-activity-detection to segment
utterances is a real, hard, hardware/OS-dependent problem — and not one an
autonomously-built package can honestly claim to have solved without a real
microphone and a human speaking into it to verify against. So `VoiceSource`
doesn't try. Instead it watches a directory for **audio files appearing on
disk**, exactly like the built-in `FilesystemSource` watches for any file —
dropped there by a phone system, a Telegram/Slack voice-message webhook, a
push-to-talk recorder, or anything else. That's a small, honest, fully
testable scope that composes cleanly with whatever actually captures audio,
which is the same "small, composable pieces" philosophy Waken itself is
built on.

## Install

```bash
pip install waken-voice
```

Needs an OpenAI API key: set `OPENAI_API_KEY` in the environment (the
default the underlying `openai` SDK reads), or pass `api_key=...` as a
keyword argument to `VoiceSource`/`VoiceOutput` — both forward unrecognized
keyword arguments straight to `openai.AsyncOpenAI(...)`.

## Usage

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

## `VoiceSource`

```python
VoiceSource(
    watch,              # directory to poll for new audio files
    target,             # name of the registered Target to dispatch to
    interval=1.0,       # poll interval, seconds
    source_name="voice",
    model="whisper-1",  # OpenAI transcription model
    **client_kwargs,    # forwarded to openai.AsyncOpenAI(...)
)
```

Polls `watch` every `interval` seconds for new files, same mechanism as the
built-in `FilesystemSource` (see
[`waken/plugins/sources/filesystem.py`](https://github.com/WakenHQ/waken/blob/main/waken/plugins/sources/filesystem.py)) —
no OS-level file-watching dependency, files present before `start()` are the
baseline and never fire. Only `.wav`, `.mp3`, `.m4a`, and `.ogg` files are
transcribed; anything else is ignored silently, since handing a non-audio
file to the transcription API is a guaranteed, pointless error rather than a
useful attempt.

## `VoiceOutput`

```python
VoiceOutput(
    output_dir,        # directory to write synthesized audio into
    voice="alloy",     # OpenAI TTS voice
    model="tts-1",     # OpenAI TTS model
    play=True,         # attempt local playback after writing the file
    **client_kwargs,   # forwarded to openai.AsyncOpenAI(...)
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
pytest
```

Tests mock `openai.AsyncOpenAI` and `VoiceOutput._play` entirely — no real
network access, API key, or audio hardware is required to run them.

## License

[MIT](https://github.com/WakenHQ/waken-voice/blob/main/LICENSE)
