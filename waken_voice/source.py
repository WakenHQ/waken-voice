"""The `VoiceSource` — dispatches one `Event` per new audio file appearing
under `watch`, transcribed to text by a pluggable `Transcriber`.

Mirrors `waken.plugins.sources.filesystem.FilesystemSource`'s shape exactly
(see docs/api-spec.md §4 Sessions and §6 Writing a Source in the main `waken`
repo) with two differences: it filters to known audio extensions before
touching a file (feeding a non-audio file to the transcription API is a
guaranteed, pointless error), and it transcribes each new file before
building the `Event`, so `payload["prompt"]` is text a `Target` can use
directly, exactly like any other prompt-shaped Event.

Scope: this is a file-drop channel — something else (a phone system, a
Telegram/Slack voice-message webhook, a push-to-talk recorder, ...) is
responsible for actually producing the audio file on disk. For live
microphone input with wake-word detection, see `MicrophoneSource` in
`waken_voice.microphone` instead.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

from waken.events import Event

from waken_voice.transcribers import OpenAIWhisperTranscriber, Transcriber

if TYPE_CHECKING:
    from waken.runtime import Runtime

_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".ogg"})


class VoiceSource:
    """Dispatches one `Event` per new audio file appearing under `watch`.

    Polls on `interval` seconds rather than an OS-level file-watching
    library, same reasoning as `FilesystemSource`: no new dependency for the
    watching mechanism itself. Files already present when `start()` runs
    are the baseline, not new arrivals, and never fire. Files without a
    recognized audio extension are skipped silently rather than sent to the
    transcriber.
    """

    def __init__(
        self,
        watch: str | Path,
        target: str,
        *,
        interval: float = 1.0,
        source_name: str = "voice",
        transcriber: Transcriber | None = None,
    ) -> None:
        self.watch = Path(watch)
        self.target = target
        self.interval = interval
        self.source_name = source_name
        self.transcriber: Transcriber = transcriber or OpenAIWhisperTranscriber()
        self._seen: set[Path] = set()
        self._task: asyncio.Task[None] | None = None

    async def start(self, runtime: Runtime) -> None:
        self.watch.mkdir(parents=True, exist_ok=True)
        self._seen = set(self.watch.iterdir())
        self._task = asyncio.create_task(self._poll(runtime))

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _poll(self, runtime: Runtime) -> None:
        while True:
            await asyncio.sleep(self.interval)
            current = set(self.watch.iterdir())
            for path in sorted(current - self._seen):
                if path.suffix.lower() not in _AUDIO_EXTENSIONS:
                    continue
                text = await self.transcriber.transcribe(path)
                event = Event(
                    source=self.source_name,
                    target=self.target,
                    payload={"prompt": text},
                    session_id=runtime.session(self.source_name, str(path)),
                )
                # Fire-and-forget, same reasoning as FilesystemSource: a slow
                # retry-with-backoff sequence for one file must not stall
                # the loop noticing the next one.
                asyncio.create_task(runtime.dispatch(event, retry=True))
            self._seen = current
