"""Tests for `VoiceSource`.

Mocks `openai.AsyncOpenAI` entirely (via `waken_voice.source.AsyncOpenAI`) —
no real network/API calls, no real audio. A fake `Transcription`-shaped
object (a `SimpleNamespace` with a `.text` attribute, matching what the real
`openai` SDK returns from `client.audio.transcriptions.create(...)`) stands
in for the API response.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from waken import Event, Response, Runtime, target_fn

from waken_voice.source import VoiceSource


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def _transcription(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


@patch("waken_voice.source.AsyncOpenAI")
async def test_new_audio_file_is_transcribed_and_dispatched(
    mock_openai_cls: AsyncMock, tmp_path: Path
) -> None:
    mock_client = mock_openai_cls.return_value
    mock_client.audio.transcriptions.create = AsyncMock(
        return_value=_transcription("hello world")
    )

    watch_dir = tmp_path / "inbox"
    received: list[Event] = []

    @target_fn
    async def echo(event: Event) -> Response:
        received.append(event)
        return Response()

    runtime = Runtime()
    runtime.target("echo", echo)
    source = VoiceSource(watch_dir, target="echo", interval=0.02, api_key="sk-test")

    await source.start(runtime)
    path = watch_dir / "message.wav"
    path.write_bytes(b"fake audio bytes")
    await asyncio.sleep(0.15)  # let the poll notice it and the dispatch task run
    await source.stop()

    assert len(received) == 1
    assert received[0].payload == {"prompt": "hello world"}
    assert received[0].source == "voice"
    assert received[0].session_id == runtime.session("voice", str(path))
    mock_client.audio.transcriptions.create.assert_awaited_once()
    _, kwargs = mock_client.audio.transcriptions.create.call_args
    assert kwargs["model"] == "whisper-1"


@patch("waken_voice.source.AsyncOpenAI")
async def test_non_audio_file_is_ignored(
    mock_openai_cls: AsyncMock, tmp_path: Path
) -> None:
    mock_client = mock_openai_cls.return_value
    mock_client.audio.transcriptions.create = AsyncMock()

    watch_dir = tmp_path / "inbox"
    received: list[Event] = []

    @target_fn
    async def echo(event: Event) -> Response:
        received.append(event)
        return Response()

    runtime = Runtime()
    runtime.target("echo", echo)
    source = VoiceSource(watch_dir, target="echo", interval=0.02, api_key="sk-test")

    await source.start(runtime)
    (watch_dir / "notes.txt").write_text("not audio")
    await asyncio.sleep(0.1)
    await source.stop()

    assert received == []
    mock_client.audio.transcriptions.create.assert_not_awaited()


@patch("waken_voice.source.AsyncOpenAI")
async def test_preexisting_file_does_not_fire(
    mock_openai_cls: AsyncMock, tmp_path: Path
) -> None:
    watch_dir = tmp_path / "inbox"
    watch_dir.mkdir()
    (watch_dir / "already-here.wav").write_bytes(b"old audio")

    mock_client = mock_openai_cls.return_value
    mock_client.audio.transcriptions.create = AsyncMock()

    received: list[Event] = []

    @target_fn
    async def echo(event: Event) -> Response:
        received.append(event)
        return Response()

    runtime = Runtime()
    runtime.target("echo", echo)
    source = VoiceSource(watch_dir, target="echo", interval=0.02, api_key="sk-test")

    await source.start(runtime)
    await asyncio.sleep(0.1)
    await source.stop()

    assert received == []
    mock_client.audio.transcriptions.create.assert_not_awaited()
