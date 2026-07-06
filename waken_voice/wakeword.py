"""Pluggable wake-word detection backends for `MicrophoneSource`.

Same shape as `Transcriber`/`Synthesizer`: a `Protocol` plus a concrete
default implementation that imports its SDK lazily, inside `__init__`
rather than at module level, so installing `waken-voice` doesn't require
`openwakeword` unless you actually use `MicrophoneSource`. See the README
for the install extra (`pip install "waken-voice[mic]"`).
"""

from __future__ import annotations

from typing import Any, Protocol


class WakeWordDetector(Protocol):
    """Looks for a wake word in one chunk of 16kHz mono 16-bit PCM audio.

    `MicrophoneSource` calls `process` once per captured frame while
    listening (not while an utterance is being recorded). Returns the name
    of the wake word that fired, or `None` if it didn't. Synchronous, not
    async: this runs local inference on a small audio frame, not I/O — same
    reasoning `GTTSSynthesizer` uses `asyncio.to_thread` for its (blocking)
    HTTP call, except here there's no I/O to move off the event loop, so a
    plain sync call is the honest shape.
    """

    def process(self, frame: bytes) -> str | None: ...


class OpenWakeWordDetector:
    """Detects wake words via `openwakeword`'s pretrained ONNX models.

    Open-source (Apache-2.0), fully offline, no API key or signup — the
    same free/no-key spirit as `GTTSSynthesizer`. Ships pretrained models
    for a handful of wake words (e.g. `"hey_jarvis"`, `"alexa"`,
    `"hey_mycroft"`); pass `wakeword_models` with paths to your own
    `.onnx`/`.tflite` models to detect something else. Model files aren't
    bundled — run `openwakeword.utils.download_models()` once before first
    use; see the README.

    Defaults to `inference_framework="onnx"` rather than openwakeword's own
    default of `"tflite"`: `tflite-runtime` has no published wheel for
    Python 3.12+ on Linux, so requiring it would make installation fail on
    exactly the versions this package targets. Pass
    `inference_framework="tflite"` yourself if you have it installed (e.g.
    on a Raspberry Pi image that ships it).
    """

    def __init__(
        self,
        *,
        wakeword_models: list[str] | None = None,
        threshold: float = 0.5,
        **model_kwargs: Any,
    ) -> None:
        from openwakeword.model import Model

        self.threshold = threshold
        model_kwargs.setdefault("inference_framework", "onnx")
        # openwakeword's own default is `[]`, not `None` — it uses `== []`
        # to decide whether to fall back to its pretrained model list, so
        # passing `None` through here would break that check.
        self._model = Model(wakeword_models=wakeword_models or [], **model_kwargs)

    def process(self, frame: bytes) -> str | None:
        import numpy as np

        samples = np.frombuffer(frame, dtype=np.int16)
        predictions = self._model.predict(samples)
        for name, score in predictions.items():
            if score > self.threshold:
                return str(name)
        return None
