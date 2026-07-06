"""The `MicrophoneSource` — dispatches one `Event` per utterance spoken
after a wake word, captured live from a microphone.

Two pluggable backends: a `WakeWordDetector` (openWakeWord by default,
fully offline) picks the utterance's start out of a continuous audio
stream, and a `Transcriber` (same one `VoiceSource` uses) turns the
recorded utterance into text once silence marks its end. See the README
for the install extra (`pip install "waken-voice[mic]"`) and the honest
caveat about this being untested against real audio hardware.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from waken.events import Event

from waken_voice.transcribers import OpenAIWhisperTranscriber, Transcriber
from waken_voice.wakeword import OpenWakeWordDetector, WakeWordDetector

if TYPE_CHECKING:
    from waken.runtime import Runtime

log = logging.getLogger("waken_voice.microphone")

_SAMPLE_WIDTH = 2  # bytes per sample, 16-bit PCM


class MicrophoneSource:
    """Listens on a microphone input, dispatches one `Event` per utterance
    spoken after the wake word.

    Driven by two states, both handled in `_handle_frame` — kept separate
    from the actual `sounddevice` stream wiring so the state machine is
    testable without real audio hardware, same reasoning as
    `VoiceOutput._play` being isolated for the same purpose:

    - **listening**: each captured frame goes to `wake_word_detector`.
      A detection switches to recording.
    - **recording**: frames are buffered. Once frames have been quiet (RMS
      amplitude below `silence_threshold`) for `silence_duration` seconds,
      or `max_utterance_duration` is reached, the buffer is written to a
      WAV file, transcribed, and dispatched as an `Event` — then it's back
      to listening for the wake word.

    **Untested against real audio hardware**, same honest caveat as
    `VoiceOutput`'s playback: this package's test suite drives
    `_handle_frame` directly with fake frames and mocks `sounddevice`
    entirely. See the README.
    """

    def __init__(
        self,
        target: str,
        *,
        source_name: str = "microphone",
        wake_word_detector: WakeWordDetector | None = None,
        transcriber: Transcriber | None = None,
        sample_rate: int = 16000,
        frame_samples: int = 1280,
        silence_threshold: float = 500.0,
        silence_duration: float = 1.0,
        max_utterance_duration: float = 15.0,
        device: int | str | None = None,
        save_recordings: str | Path | None = None,
    ) -> None:
        self.target = target
        self.source_name = source_name
        self.wake_word_detector: WakeWordDetector = (
            wake_word_detector or OpenWakeWordDetector()
        )
        self.transcriber: Transcriber = transcriber or OpenAIWhisperTranscriber()
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_utterance_duration = max_utterance_duration
        self.device = device
        self.save_recordings = Path(save_recordings) if save_recordings else None

        self._recording = False
        self._buffer = bytearray()
        self._silent_frames = 0
        self._recorded_frames = 0
        self._frame_duration = frame_samples / sample_rate

        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._stream: Any | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self, runtime: Runtime) -> None:
        import sounddevice as sd

        if self.save_recordings is not None:
            self.save_recordings.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()

        def _callback(indata: Any, frames: int, time: Any, status: Any) -> None:
            if status:
                log.warning("microphone input status: %s", status)
            loop.call_soon_threadsafe(self._queue.put_nowait, bytes(indata))

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            device=self.device,
            callback=_callback,
        )
        self._stream.start()
        self._task = asyncio.create_task(self._consume(runtime))

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def _consume(self, runtime: Runtime) -> None:
        while True:
            frame = await self._queue.get()
            await self._handle_frame(runtime, frame)

    async def _handle_frame(self, runtime: Runtime, frame: bytes) -> None:
        if not self._recording:
            name = self.wake_word_detector.process(frame)
            if name is not None:
                log.info("wake word %r detected, recording utterance", name)
                self._recording = True
                self._buffer = bytearray()
                self._silent_frames = 0
                self._recorded_frames = 0
            return

        self._buffer.extend(frame)
        self._recorded_frames += 1
        if _rms(frame) < self.silence_threshold:
            self._silent_frames += 1
        else:
            self._silent_frames = 0

        elapsed = self._recorded_frames * self._frame_duration
        silence_elapsed = self._silent_frames * self._frame_duration
        if (
            silence_elapsed >= self.silence_duration
            or elapsed >= self.max_utterance_duration
        ):
            await self._finish_utterance(runtime)

    async def _finish_utterance(self, runtime: Runtime) -> None:
        self._recording = False
        buffer, self._buffer = self._buffer, bytearray()

        if self.save_recordings is not None:
            path = self.save_recordings / f"{uuid4().hex}.wav"
            _write_wav(path, bytes(buffer), self.sample_rate)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = Path(tmp.name)
            _write_wav(path, bytes(buffer), self.sample_rate)

        try:
            text = await self.transcriber.transcribe(path)
        finally:
            if self.save_recordings is None:
                path.unlink(missing_ok=True)

        if not text.strip():
            return

        event = Event(
            source=self.source_name,
            target=self.target,
            payload={"prompt": text},
            session_id=runtime.session(self.source_name, uuid4().hex),
        )
        # Fire-and-forget, same reasoning as VoiceSource/FilesystemSource: a
        # slow retry-with-backoff sequence for one utterance must not stall
        # the loop noticing the next wake word.
        asyncio.create_task(runtime.dispatch(event, retry=True))


def _rms(frame: bytes) -> float:
    import numpy as np

    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)))


def _write_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(_SAMPLE_WIDTH)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
