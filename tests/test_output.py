"""Tests for `VoiceOutput`.

Mocks `openai.AsyncOpenAI` entirely (via `waken_voice.output.AsyncOpenAI`)
and mocks `_play` directly rather than the subprocess machinery underneath
it — playback itself is untested against real audio hardware by design (see
the README), so tests only assert *that* `_play` was invoked with the right
path, never what it actually does.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from waken import Event, Response

from waken_voice.output import VoiceOutput


def _speech(content: bytes) -> SimpleNamespace:
    return SimpleNamespace(content=content)


def _event() -> Event:
    return Event(source="api", target="echo", payload={})


@patch("waken_voice.output.AsyncOpenAI")
async def test_text_is_synthesized_and_written_to_file(
    mock_openai_cls: AsyncMock, tmp_path: Path
) -> None:
    mock_client = mock_openai_cls.return_value
    mock_client.audio.speech.create = AsyncMock(return_value=_speech(b"fake mp3 bytes"))

    output = VoiceOutput(tmp_path / "out", play=False, api_key="sk-test")
    output._play = AsyncMock()  # type: ignore[method-assign]

    event = _event()
    await output.deliver(event, Response(text="hello there"))

    expected_path = tmp_path / "out" / f"{event.event_id}.mp3"
    assert expected_path.read_bytes() == b"fake mp3 bytes"
    mock_client.audio.speech.create.assert_awaited_once()
    _, kwargs = mock_client.audio.speech.create.call_args
    assert kwargs["input"] == "hello there"
    assert kwargs["model"] == "tts-1"
    assert kwargs["voice"] == "alloy"


@patch("waken_voice.output.AsyncOpenAI")
async def test_play_false_never_calls_play(
    mock_openai_cls: AsyncMock, tmp_path: Path
) -> None:
    mock_client = mock_openai_cls.return_value
    mock_client.audio.speech.create = AsyncMock(return_value=_speech(b"bytes"))

    output = VoiceOutput(tmp_path / "out", play=False, api_key="sk-test")
    output._play = AsyncMock()  # type: ignore[method-assign]

    await output.deliver(_event(), Response(text="hi"))

    output._play.assert_not_awaited()


@patch("waken_voice.output.AsyncOpenAI")
async def test_play_true_calls_play_with_written_path(
    mock_openai_cls: AsyncMock, tmp_path: Path
) -> None:
    mock_client = mock_openai_cls.return_value
    mock_client.audio.speech.create = AsyncMock(return_value=_speech(b"bytes"))

    output = VoiceOutput(tmp_path / "out", play=True, api_key="sk-test")
    output._play = AsyncMock()  # type: ignore[method-assign]

    event = _event()
    await output.deliver(event, Response(text="hi"))

    expected_path = tmp_path / "out" / f"{event.event_id}.mp3"
    output._play.assert_awaited_once_with(expected_path)


@patch("waken_voice.output.AsyncOpenAI")
async def test_empty_or_none_text_delivers_nothing(
    mock_openai_cls: AsyncMock, tmp_path: Path
) -> None:
    mock_client = mock_openai_cls.return_value
    mock_client.audio.speech.create = AsyncMock()

    output = VoiceOutput(tmp_path / "out", api_key="sk-test")
    output._play = AsyncMock()  # type: ignore[method-assign]

    await output.deliver(_event(), Response(text=None))
    await output.deliver(_event(), Response(text=""))

    assert not (tmp_path / "out").exists()
    mock_client.audio.speech.create.assert_not_awaited()
    output._play.assert_not_awaited()
