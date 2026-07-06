"""Tests for `MicrophoneSource`.

Drives `_handle_frame` directly with synthetic PCM frames rather than a
real audio stream — recording/wake-word detection over real microphone
hardware is untested by design, same as `VoiceOutput._play`; see the
README. `start`/`stop`'s wiring to `sounddevice.InputStream` is covered
separately below with the SDK mocked.
"""

import asyncio
import struct
from pathlib import Path
from unittest.mock import Mock, patch

from waken import Event, Response, Runtime, target_fn

from waken_voice.microphone import MicrophoneSource
from waken_voice.wakeword import OpenWakeWordDetector

SAMPLE_RATE = 16000
FRAME_SAMPLES = 160  # small frame so tests need few iterations
FRAME_DURATION = FRAME_SAMPLES / SAMPLE_RATE


def _frame(amplitude: int, samples: int = FRAME_SAMPLES) -> bytes:
    return struct.pack(f"<{samples}h", *([amplitude] * samples))


class _FakeWakeWordDetector:
    """Triggers "hey_test" on the `trigger_on`-th call to `process`."""

    def __init__(self, trigger_on: int = 0) -> None:
        self.trigger_on = trigger_on
        self.calls = 0

    def process(self, frame: bytes) -> str | None:
        call_index = self.calls
        self.calls += 1
        return "hey_test" if call_index == self.trigger_on else None


class _FakeTranscriber:
    def __init__(self, text: str = "hello world") -> None:
        self.text = text
        self.calls: list[Path] = []

    async def transcribe(self, path: Path) -> str:
        assert path.exists()
        self.calls.append(path)
        return self.text


def _make_source(
    **kwargs: object,
) -> tuple[MicrophoneSource, Runtime, list[Event]]:
    received: list[Event] = []

    @target_fn
    async def echo(event: Event) -> Response:
        received.append(event)
        return Response()

    runtime = Runtime()
    runtime.target("echo", echo)

    source = MicrophoneSource(
        target="echo",
        sample_rate=SAMPLE_RATE,
        frame_samples=FRAME_SAMPLES,
        **kwargs,  # type: ignore[arg-type]
    )
    return source, runtime, received


async def test_utterance_is_recorded_transcribed_and_dispatched() -> None:
    detector = _FakeWakeWordDetector(trigger_on=0)
    transcriber = _FakeTranscriber("turn on the lights")
    source, runtime, received = _make_source(
        wake_word_detector=detector,
        transcriber=transcriber,
        silence_duration=FRAME_DURATION * 2,
    )

    await source._handle_frame(runtime, _frame(0))  # the wake word itself
    await source._handle_frame(runtime, _frame(5000))  # loud speech
    await source._handle_frame(runtime, _frame(0))  # silence starts...
    await source._handle_frame(runtime, _frame(0))  # ...long enough to end it
    await asyncio.sleep(0.05)  # let the fire-and-forget dispatch task run

    assert len(received) == 1
    assert received[0].payload == {"prompt": "turn on the lights"}
    assert received[0].source == "microphone"
    assert len(transcriber.calls) == 1
    assert not source._recording  # back to listening for the next wake word


async def test_no_wake_word_never_records_or_transcribes() -> None:
    detector = _FakeWakeWordDetector(trigger_on=999)  # never triggers
    transcriber = _FakeTranscriber()
    source, runtime, received = _make_source(
        wake_word_detector=detector, transcriber=transcriber
    )

    for _ in range(5):
        await source._handle_frame(runtime, _frame(5000))

    assert received == []
    assert transcriber.calls == []
    assert not source._recording


async def test_max_utterance_duration_cuts_off_long_speech() -> None:
    detector = _FakeWakeWordDetector(trigger_on=0)
    transcriber = _FakeTranscriber("long ramble")
    source, runtime, received = _make_source(
        wake_word_detector=detector,
        transcriber=transcriber,
        silence_duration=10.0,  # never triggers via silence in this test
        max_utterance_duration=FRAME_DURATION * 2,
    )

    await source._handle_frame(runtime, _frame(0))  # wake word
    await source._handle_frame(runtime, _frame(5000))  # 1 frame of speech
    await source._handle_frame(runtime, _frame(5000))  # 2nd hits the cap
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert len(transcriber.calls) == 1
    assert not source._recording


async def test_empty_transcription_is_not_dispatched() -> None:
    detector = _FakeWakeWordDetector(trigger_on=0)
    transcriber = _FakeTranscriber("   ")  # blank after strip()
    source, runtime, received = _make_source(
        wake_word_detector=detector,
        transcriber=transcriber,
        silence_duration=FRAME_DURATION,
    )

    await source._handle_frame(runtime, _frame(0))  # wake word
    await source._handle_frame(runtime, _frame(5000))
    await source._handle_frame(runtime, _frame(0))  # ends the utterance
    await asyncio.sleep(0.05)

    assert received == []
    assert len(transcriber.calls) == 1


async def test_save_recordings_keeps_the_wav_file(tmp_path: Path) -> None:
    detector = _FakeWakeWordDetector(trigger_on=0)
    transcriber = _FakeTranscriber("kept")
    source, runtime, received = _make_source(
        wake_word_detector=detector,
        transcriber=transcriber,
        silence_duration=FRAME_DURATION,
        save_recordings=tmp_path / "recordings",
    )
    source.save_recordings.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]

    await source._handle_frame(runtime, _frame(0))
    await source._handle_frame(runtime, _frame(5000))
    await source._handle_frame(runtime, _frame(0))
    await asyncio.sleep(0.05)

    saved = list((tmp_path / "recordings").glob("*.wav"))
    assert len(saved) == 1
    assert transcriber.calls[0] == saved[0]
    assert len(received) == 1


@patch("openwakeword.model.Model")
def test_default_wake_word_detector_is_openwakeword(mock_model_cls: Mock) -> None:
    """No `wake_word_detector=` given falls back to `OpenWakeWordDetector`."""
    source, _, _ = _make_source(transcriber=_FakeTranscriber())

    assert isinstance(source.wake_word_detector, OpenWakeWordDetector)


@patch("sounddevice.InputStream")
async def test_start_opens_input_stream_and_stop_closes_it(
    mock_stream_cls: Mock,
) -> None:
    detector = _FakeWakeWordDetector()
    source, runtime, _ = _make_source(
        wake_word_detector=detector, transcriber=_FakeTranscriber()
    )

    await source.start(runtime)

    mock_stream_cls.assert_called_once()
    _, kwargs = mock_stream_cls.call_args
    assert kwargs["samplerate"] == SAMPLE_RATE
    assert kwargs["channels"] == 1
    assert kwargs["blocksize"] == FRAME_SAMPLES
    mock_stream_cls.return_value.start.assert_called_once()

    await source.stop()

    mock_stream_cls.return_value.stop.assert_called_once()
    mock_stream_cls.return_value.close.assert_called_once()
