"""The `VoiceOutput` — synthesizes a `Response`'s text to speech via OpenAI's
TTS API and writes it to disk, optionally playing it back.

See docs/api-spec.md §7 (Writing an Output) in the main `waken` repo for the
conceptual model.
"""

from __future__ import annotations

import asyncio
import logging
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from waken.events import Event
    from waken.responses import Response

log = logging.getLogger("waken_voice.output")


class VoiceOutput:
    """Synthesizes `response.text` to speech and writes it next to `output_dir`.

    A `Response` with no `text` (e.g. a `Target` that only returned `files`)
    delivers nothing — skipped silently rather than sending an empty string
    to the TTS API, which would error.

    Playback (`play=True`, the default) is **best-effort and untested
    against real audio hardware/OS combinations.** `_play` shells out to a
    platform audio player (`afplay` on macOS, `paplay` on Linux, an
    unverified fallback elsewhere) via `asyncio.create_subprocess_exec`
    after the file is already safely written to disk — a missing/failing
    player is logged and swallowed, not raised, since the primary contract
    (write the audio somewhere) already succeeded by that point. This
    package's own test suite mocks `_play` entirely rather than pretending
    to verify it; see the README for the full caveat.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        voice: str = "alloy",
        model: str = "tts-1",
        play: bool = True,
        **client_kwargs: Any,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.voice = voice
        self.model = model
        self.play = play
        self._client = AsyncOpenAI(**client_kwargs)

    async def deliver(self, event: Event, response: Response) -> None:
        if not response.text:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        audio_bytes = await self._synthesize(response.text)
        path = self.output_dir / f"{event.event_id}.mp3"
        path.write_bytes(audio_bytes)
        if self.play:
            await self._play(path)

    async def _synthesize(self, text: str) -> bytes:
        speech = await self._client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
        )
        return speech.content

    async def _play(self, path: Path) -> None:
        """Best-effort playback via a platform audio player.

        Kept small and separate from `deliver`/`_synthesize` on purpose: this
        is the one piece of this package that genuinely cannot be verified
        without real audio hardware, so it's isolated to make mocking it
        trivial in tests (and to make it easy to swap for something else —
        a different player, a Bluetooth sink, an HTTP callback — without
        touching the synthesis/delivery logic above it).
        """
        system = platform.system()
        if system == "Darwin":
            command: tuple[str, ...] = ("afplay", str(path))
        elif system == "Linux":
            command = ("paplay", str(path))
        else:
            # No player verified for this platform (e.g. Windows) — best
            # guess, not a guarantee. See the README.
            command = ("aplay", str(path))
        try:
            process = await asyncio.create_subprocess_exec(*command)
            await process.wait()
        except (FileNotFoundError, OSError):
            log.warning(
                "voice playback failed (player %r unavailable?) for %s",
                command[0],
                path,
            )
