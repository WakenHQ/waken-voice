"""Tests for `VoiceOutput`.

Uses a fake `Synthesizer` (any object with an async `synthesize` method) —
the real synthesizer implementations (`OpenAITTSSynthesizer`,
`GTTSSynthesizer`) have their own tests in test_synthesizers.py. `_play` is
mocked directly rather than the subprocess machinery underneath it —
playback itself is untested against real audio hardware by design (see the
README), so tests only assert *that* `_play` was invoked with the right
path, never what it actually does.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from waken import Event, Response

from waken_voice.output import VoiceOutput
from waken_voice.synthesizers import OpenAITTSSynthesizer


def _event() -> Event:
    return Event(source="api", target="echo", payload={})


class _FakeSynthesizer:
    def __init__(self, audio: bytes = b"fake audio bytes") -> None:
        self.audio = audio
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return self.audio


async def test_text_is_synthesized_and_written_to_file(tmp_path: Path) -> None:
    synthesizer = _FakeSynthesizer(b"fake mp3 bytes")
    output = VoiceOutput(tmp_path / "out", synthesizer=synthesizer, play=False)
    output._play = AsyncMock()  # type: ignore[method-assign]

    event = _event()
    await output.deliver(event, Response(text="hello there"))

    expected_path = tmp_path / "out" / f"{event.event_id}.mp3"
    assert expected_path.read_bytes() == b"fake mp3 bytes"
    assert synthesizer.calls == ["hello there"]


async def test_play_false_never_calls_play(tmp_path: Path) -> None:
    synthesizer = _FakeSynthesizer()
    output = VoiceOutput(tmp_path / "out", synthesizer=synthesizer, play=False)
    output._play = AsyncMock()  # type: ignore[method-assign]

    await output.deliver(_event(), Response(text="hi"))

    output._play.assert_not_awaited()


async def test_play_true_calls_play_with_written_path(tmp_path: Path) -> None:
    synthesizer = _FakeSynthesizer()
    output = VoiceOutput(tmp_path / "out", synthesizer=synthesizer, play=True)
    output._play = AsyncMock()  # type: ignore[method-assign]

    event = _event()
    await output.deliver(event, Response(text="hi"))

    expected_path = tmp_path / "out" / f"{event.event_id}.mp3"
    output._play.assert_awaited_once_with(expected_path)


async def test_empty_or_none_text_delivers_nothing(tmp_path: Path) -> None:
    synthesizer = _FakeSynthesizer()
    output = VoiceOutput(tmp_path / "out", synthesizer=synthesizer)
    output._play = AsyncMock()  # type: ignore[method-assign]

    await output.deliver(_event(), Response(text=None))
    await output.deliver(_event(), Response(text=""))

    assert not (tmp_path / "out").exists()
    assert synthesizer.calls == []
    output._play.assert_not_awaited()


@patch("openai.AsyncOpenAI")
async def test_default_synthesizer_is_openai(
    mock_openai_cls: AsyncMock, tmp_path: Path
) -> None:
    """No `synthesizer=` given falls back to `OpenAITTSSynthesizer`."""
    output = VoiceOutput(tmp_path / "out")

    assert isinstance(output.synthesizer, OpenAITTSSynthesizer)
