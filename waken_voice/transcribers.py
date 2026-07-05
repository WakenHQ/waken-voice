"""Pluggable speech-to-text backends for `VoiceSource`.

Each transcriber imports its own SDK lazily, inside `__init__` rather than
at module level — so installing `waken-voice` doesn't require every
provider's client library, only whichever one you actually instantiate.
See the README for the install extras (`pip install "waken-voice[openai]"`
/ `[groq]`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Transcriber(Protocol):
    """Turns an audio file into text.

    `VoiceSource` calls `transcribe` once per new file. Anything with a
    matching async method works — no base class required, same
    name-keyed-object philosophy as the rest of Waken.
    """

    async def transcribe(self, path: Path) -> str: ...


class OpenAIWhisperTranscriber:
    """Transcribes via OpenAI's Whisper-backed audio API.

    The original, still-default backend — unchanged behavior from before
    `Transcriber` existed as a separate concept.
    """

    def __init__(self, *, model: str = "whisper-1", **client_kwargs: Any) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self._client = AsyncOpenAI(**client_kwargs)

    async def transcribe(self, path: Path) -> str:
        with path.open("rb") as audio_file:
            transcription = await self._client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
            )
        return transcription.text


class GroqWhisperTranscriber:
    """Transcribes via Groq's Whisper-backed audio API.

    Same request/response shape as OpenAI's — Groq's SDK mirrors it
    deliberately — just routed to Groq's inference infrastructure instead,
    with `whisper-large-v3` as the default model. Groq has no TTS
    counterpart; pair this with `GTTSSynthesizer` or `OpenAITTSSynthesizer`
    for `VoiceOutput`.
    """

    def __init__(
        self, *, model: str = "whisper-large-v3", **client_kwargs: Any
    ) -> None:
        from groq import AsyncGroq

        self.model = model
        self._client = AsyncGroq(**client_kwargs)

    async def transcribe(self, path: Path) -> str:
        with path.open("rb") as audio_file:
            transcription = await self._client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
            )
        return transcription.text
