"""Tests for the pluggable `WakeWordDetector` implementations.

`openwakeword.model.Model` is patched at its defining module rather than as
a `waken_voice.wakeword` attribute, since `OpenWakeWordDetector` imports it
lazily inside `__init__` — only whichever detector you actually use needs
to be installed.
"""

from unittest.mock import Mock, patch

from waken_voice.wakeword import OpenWakeWordDetector


@patch("openwakeword.model.Model")
def test_detects_wake_word_above_threshold(mock_model_cls: Mock) -> None:
    mock_model_cls.return_value.predict.return_value = {"hey_jarvis": 0.9}

    detector = OpenWakeWordDetector(threshold=0.5)
    result = detector.process(b"\x00\x00" * 1280)

    assert result == "hey_jarvis"


@patch("openwakeword.model.Model")
def test_no_detection_below_threshold(mock_model_cls: Mock) -> None:
    mock_model_cls.return_value.predict.return_value = {"hey_jarvis": 0.1}

    detector = OpenWakeWordDetector(threshold=0.5)
    result = detector.process(b"\x00\x00" * 1280)

    assert result is None


@patch("openwakeword.model.Model")
def test_first_wake_word_above_threshold_wins(mock_model_cls: Mock) -> None:
    mock_model_cls.return_value.predict.return_value = {
        "hey_jarvis": 0.1,
        "alexa": 0.8,
    }

    detector = OpenWakeWordDetector(threshold=0.5)
    result = detector.process(b"\x00\x00" * 1280)

    assert result == "alexa"


@patch("openwakeword.model.Model")
def test_passes_wakeword_models_and_kwargs_through(mock_model_cls: Mock) -> None:
    OpenWakeWordDetector(wakeword_models=["custom.onnx"], vad_threshold=0.5)

    _, kwargs = mock_model_cls.call_args
    assert kwargs["wakeword_models"] == ["custom.onnx"]
    assert kwargs["vad_threshold"] == 0.5
