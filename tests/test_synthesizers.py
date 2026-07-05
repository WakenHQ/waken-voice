"""Tests for the pluggable `Synthesizer` implementations.

`openai.AsyncOpenAI` and `gtts.gTTS` are patched at their defining modules
rather than as `waken_voice.synthesizers` attributes, since these classes
import their SDK lazily inside `__init__`/the synthesis call — only
whichever provider you actually use needs to be installed.
"""

from types import SimpleNamespace
from typing import IO
from unittest.mock import AsyncMock, Mock, patch

from waken_voice.synthesizers import GTTSSynthesizer, OpenAITTSSynthesizer


def _speech(content: bytes) -> SimpleNamespace:
    return SimpleNamespace(content=content)


@patch("openai.AsyncOpenAI")
async def test_openai_synthesizer_returns_bytes(mock_openai_cls: AsyncMock) -> None:
    mock_client = mock_openai_cls.return_value
    mock_client.audio.speech.create = AsyncMock(return_value=_speech(b"fake mp3"))

    synthesizer = OpenAITTSSynthesizer(api_key="sk-test")
    audio = await synthesizer.synthesize("hello there")

    assert audio == b"fake mp3"
    _, kwargs = mock_client.audio.speech.create.call_args
    assert kwargs["input"] == "hello there"
    assert kwargs["model"] == "tts-1"
    assert kwargs["voice"] == "alloy"


@patch("gtts.gTTS")
async def test_gtts_synthesizer_returns_bytes(mock_gtts_cls: Mock) -> None:
    def _fake_write_to_fp(fp: IO[bytes]) -> None:
        fp.write(b"fake mp3 from gtts")

    mock_gtts_cls.return_value.write_to_fp.side_effect = _fake_write_to_fp

    synthesizer = GTTSSynthesizer(lang="en")
    audio = await synthesizer.synthesize("hello there")

    assert audio == b"fake mp3 from gtts"
    _, kwargs = mock_gtts_cls.call_args
    assert kwargs["text"] == "hello there"
    assert kwargs["lang"] == "en"


@patch("gtts.gTTS")
async def test_gtts_synthesizer_passes_through_extra_kwargs(
    mock_gtts_cls: Mock,
) -> None:
    mock_gtts_cls.return_value.write_to_fp.side_effect = lambda fp: fp.write(b"x")

    synthesizer = GTTSSynthesizer(lang="pt", slow=True)
    await synthesizer.synthesize("ola")

    _, kwargs = mock_gtts_cls.call_args
    assert kwargs["lang"] == "pt"
    assert kwargs["slow"] is True
