"""Tests for `VoiceSource`.

Uses a fake `Transcriber` (any object with an async `transcribe` method) —
the real transcriber implementations (`OpenAIWhisperTranscriber`,
`GroqWhisperTranscriber`) have their own tests in test_transcribers.py.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from waken import Event, Response, Runtime, target_fn

from waken_voice.source import VoiceSource
from waken_voice.transcribers import OpenAIWhisperTranscriber


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


class _FakeTranscriber:
    def __init__(self, text: str = "hello world") -> None:
        self.text = text
        self.calls: list[Path] = []

    async def transcribe(self, path: Path) -> str:
        self.calls.append(path)
        return self.text


async def test_new_audio_file_is_transcribed_and_dispatched(tmp_path: Path) -> None:
    watch_dir = tmp_path / "inbox"
    received: list[Event] = []

    @target_fn
    async def echo(event: Event) -> Response:
        received.append(event)
        return Response()

    runtime = Runtime()
    runtime.target("echo", echo)
    transcriber = _FakeTranscriber("hello world")
    source = VoiceSource(
        watch_dir, target="echo", interval=0.02, transcriber=transcriber
    )

    await source.start(runtime)
    path = watch_dir / "message.wav"
    path.write_bytes(b"fake audio bytes")
    await asyncio.sleep(0.15)  # let the poll notice it and the dispatch task run
    await source.stop()

    assert len(received) == 1
    assert received[0].payload == {"prompt": "hello world"}
    assert received[0].source == "voice"
    assert received[0].session_id == runtime.session("voice", str(path))
    assert transcriber.calls == [path]


async def test_non_audio_file_is_ignored(tmp_path: Path) -> None:
    watch_dir = tmp_path / "inbox"
    received: list[Event] = []

    @target_fn
    async def echo(event: Event) -> Response:
        received.append(event)
        return Response()

    runtime = Runtime()
    runtime.target("echo", echo)
    transcriber = _FakeTranscriber()
    source = VoiceSource(
        watch_dir, target="echo", interval=0.02, transcriber=transcriber
    )

    await source.start(runtime)
    (watch_dir / "notes.txt").write_text("not audio")
    await asyncio.sleep(0.1)
    await source.stop()

    assert received == []
    assert transcriber.calls == []


async def test_preexisting_file_does_not_fire(tmp_path: Path) -> None:
    watch_dir = tmp_path / "inbox"
    watch_dir.mkdir()
    (watch_dir / "already-here.wav").write_bytes(b"old audio")

    received: list[Event] = []

    @target_fn
    async def echo(event: Event) -> Response:
        received.append(event)
        return Response()

    runtime = Runtime()
    runtime.target("echo", echo)
    transcriber = _FakeTranscriber()
    source = VoiceSource(
        watch_dir, target="echo", interval=0.02, transcriber=transcriber
    )

    await source.start(runtime)
    await asyncio.sleep(0.1)
    await source.stop()

    assert received == []
    assert transcriber.calls == []


@patch("openai.AsyncOpenAI")
async def test_default_transcriber_is_openai(
    mock_openai_cls: AsyncMock, tmp_path: Path
) -> None:
    """No `transcriber=` given falls back to `OpenAIWhisperTranscriber`."""
    source = VoiceSource(tmp_path / "inbox", target="echo")

    assert isinstance(source.transcriber, OpenAIWhisperTranscriber)
