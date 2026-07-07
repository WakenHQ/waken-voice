"""Tests for the small stand-alone helpers in `waken_voice.util`."""

import struct
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from waken_voice.util import describe_portaudio_error, resolve_device, rms, write_wav


def test_resolve_device_none_stays_none() -> None:
    assert resolve_device(None) is None


def test_resolve_device_digit_string_becomes_int() -> None:
    assert resolve_device("3") == 3


def test_resolve_device_name_stays_string() -> None:
    assert resolve_device("pulse") == "pulse"


def test_rms_of_silence_is_zero() -> None:
    assert rms(b"\x00\x00" * 100) == 0.0


def test_rms_of_empty_frame_is_zero() -> None:
    assert rms(b"") == 0.0


def test_rms_of_constant_amplitude_matches_that_amplitude() -> None:
    frame = struct.pack("<4h", 1000, -1000, 1000, -1000)
    assert rms(frame) == 1000.0


def test_write_wav_round_trips_pcm_bytes(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    pcm = struct.pack("<4h", 100, 200, 300, 400)

    write_wav(path, pcm, sample_rate=16000)

    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.readframes(wav_file.getnframes()) == pcm


@patch("sounddevice.query_devices")
def test_describe_portaudio_error_lists_input_devices_and_rerun_command(
    mock_query_devices: Mock,
) -> None:
    mock_query_devices.return_value = [
        {"name": "default", "max_input_channels": 0},
        {"name": "pulse", "max_input_channels": 32},
    ]

    message = describe_portaudio_error("WAKEN_MIC_DEVICE=pulse uv run script.py")

    assert "[1] pulse" in message
    assert "[0] default" not in message
    assert "WAKEN_MIC_DEVICE=pulse uv run script.py" in message
