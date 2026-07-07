"""Small stand-alone helpers with no state and no pluggable-backend
concept of their own — used internally by other waken_voice modules and
safe for consumers (e.g. a sample script embedding `MicrophoneSource`) to
import directly. Deliberately not re-exported from `__init__.py`: that
facade's `__all__` is reserved for the Source/Output/Synthesizer/
Transcriber/WakeWordDetector Protocol + concrete-implementation pairs.
"""

from __future__ import annotations

import wave
from pathlib import Path


def resolve_device(value: str | None) -> int | str | None:
    """Parses an env var (e.g. `WAKEN_MIC_DEVICE`) into whatever
    `sounddevice`'s `device=` kwarg expects: an int index, a device name,
    or `None` (its own default) if unset.
    """
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def rms(frame: bytes) -> float:
    """RMS (root-mean-square) loudness of one frame of mono 16-bit PCM
    audio — used to tell speech from silence. Imports numpy lazily, same
    reasoning as the pluggable backends' lazy SDK imports: numpy is part
    of the `mic` extra, not a base dependency, so importing this module
    shouldn't require it unless this function is actually called.
    """
    import numpy as np

    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)))


def write_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    """Writes mono 16-bit PCM bytes to a `.wav` file at `path`."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # int16
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def describe_portaudio_error(rerun_command: str) -> str:
    """Explains a `sounddevice.PortAudioError` raised while starting an
    input stream, and lists the available input devices.

    Most commonly hit on WSL, where PortAudio's ALSA backend defaults to
    a device with no real hardware behind it — the actual microphone is
    bridged in over PulseAudio (WSLg) instead, typically as a device
    named "pulse". `rerun_command` is the exact command line the caller
    should print for the reader to copy-paste with a working device
    slotted in via `WAKEN_MIC_DEVICE`.
    """
    import sounddevice as sd

    lines = [
        "On WSL this is almost always ALSA opening a device with no real",
        "hardware behind it — the actual mic is bridged in over PulseAudio",
        "(WSLg), not raw ALSA. Available input devices:",
    ]
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            lines.append(f"  [{index}] {info['name']}")
    lines.append(
        f"Pick one whose name contains 'pulse', then rerun with:\n  {rerun_command}"
    )
    return "\n".join(lines)
