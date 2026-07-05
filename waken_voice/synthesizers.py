"""Pluggable text-to-speech backends for `VoiceOutput`.

Each synthesizer imports its own SDK lazily, inside `__init__` (or the
synthesis call itself) rather than at module level — so installing
`waken-voice` doesn't require every provider's client library, only
whichever one you actually instantiate. See the README for the install
extras (`pip install "waken-voice[openai]"` / `[gtts]`).
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol


class Synthesizer(Protocol):
    """Turns text into audio bytes.

    `VoiceOutput` calls `synthesize` once per delivered `Response`. Anything
    with a matching async method works — no base class required, same
    name-keyed-object philosophy as the rest of Waken.
    """

    async def synthesize(self, text: str) -> bytes: ...


class OpenAITTSSynthesizer:
    """Synthesizes via OpenAI's TTS API.

    The original, still-default backend — unchanged behavior from before
    `Synthesizer` existed as a separate concept.
    """

    def __init__(
        self, *, voice: str = "alloy", model: str = "tts-1", **client_kwargs: Any
    ) -> None:
        from openai import AsyncOpenAI

        self.voice = voice
        self.model = model
        self._client = AsyncOpenAI(**client_kwargs)

    async def synthesize(self, text: str) -> bytes:
        speech = await self._client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
        )
        return speech.content


class GTTSSynthesizer:
    """Synthesizes via `gTTS`, which wraps Google Translate's undocumented
    TTS endpoint.

    Free and needs no API key, but it's a reverse-engineered API with no
    uptime or rate-limit contract, and a small, fixed set of accented
    voices per language rather than named-voice selection like OpenAI's
    `alloy`/`nova`/... `gTTS` itself is synchronous, so this runs it in a
    thread via `asyncio.to_thread` to avoid blocking the event loop.
    """

    def __init__(self, *, lang: str = "en", **gtts_kwargs: Any) -> None:
        self.lang = lang
        self._gtts_kwargs = gtts_kwargs

    async def synthesize(self, text: str) -> bytes:
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        from io import BytesIO

        from gtts import gTTS

        buffer = BytesIO()
        gTTS(text=text, lang=self.lang, **self._gtts_kwargs).write_to_fp(buffer)
        return buffer.getvalue()
