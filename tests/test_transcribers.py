"""Tests for the pluggable `Transcriber` implementations.

Each provider's SDK client is patched at its defining module (`openai.
AsyncOpenAI`, `groq.AsyncGroq`) rather than as a `waken_voice.transcribers`
attribute, since these classes import their SDK lazily inside `__init__` —
only whichever provider you actually use needs to be installed.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from waken_voice.transcribers import GroqWhisperTranscriber, OpenAIWhisperTranscriber


def _transcription(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


@patch("openai.AsyncOpenAI")
async def test_openai_transcriber_returns_text(
    mock_openai_cls: AsyncMock, tmp_path: Path
) -> None:
    mock_client = mock_openai_cls.return_value
    mock_client.audio.transcriptions.create = AsyncMock(
        return_value=_transcription("hello from openai")
    )
    path = tmp_path / "message.wav"
    path.write_bytes(b"fake audio")

    transcriber = OpenAIWhisperTranscriber(api_key="sk-test")
    text = await transcriber.transcribe(path)

    assert text == "hello from openai"
    _, kwargs = mock_client.audio.transcriptions.create.call_args
    assert kwargs["model"] == "whisper-1"


@patch("groq.AsyncGroq")
async def test_groq_transcriber_returns_text(
    mock_groq_cls: AsyncMock, tmp_path: Path
) -> None:
    mock_client = mock_groq_cls.return_value
    mock_client.audio.transcriptions.create = AsyncMock(
        return_value=_transcription("hello from groq")
    )
    path = tmp_path / "message.wav"
    path.write_bytes(b"fake audio")

    transcriber = GroqWhisperTranscriber(api_key="gsk-test")
    text = await transcriber.transcribe(path)

    assert text == "hello from groq"
    _, kwargs = mock_client.audio.transcriptions.create.call_args
    assert kwargs["model"] == "whisper-large-v3"


@patch("groq.AsyncGroq")
async def test_groq_transcriber_accepts_custom_model(
    mock_groq_cls: AsyncMock, tmp_path: Path
) -> None:
    mock_client = mock_groq_cls.return_value
    mock_client.audio.transcriptions.create = AsyncMock(
        return_value=_transcription("ok")
    )
    path = tmp_path / "message.wav"
    path.write_bytes(b"fake audio")

    transcriber = GroqWhisperTranscriber(
        model="distil-whisper-large-v3-en", api_key="gsk-test"
    )
    await transcriber.transcribe(path)

    _, kwargs = mock_client.audio.transcriptions.create.call_args
    assert kwargs["model"] == "distil-whisper-large-v3-en"
