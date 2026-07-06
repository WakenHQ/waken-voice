"""Voice Source/Output for Waken.

`VoiceSource` turns audio files dropped on disk into `Event`s via a
pluggable `Transcriber` (OpenAI Whisper by default, or Groq's
Whisper-backed API); `MicrophoneSource` does the same for utterances spoken
live after a wake word, via a pluggable `WakeWordDetector` (openWakeWord by
default) plus the same `Transcriber`. `VoiceOutput` turns a `Response`'s
text into audio files on disk via a pluggable `Synthesizer` (OpenAI TTS by
default, or `gTTS`), optionally playing them back through a platform audio
player.

See the package README for the install extras and the honest caveats
around anything touching real audio hardware.
"""

from waken_voice.microphone import MicrophoneSource
from waken_voice.output import VoiceOutput
from waken_voice.source import VoiceSource
from waken_voice.synthesizers import GTTSSynthesizer, OpenAITTSSynthesizer, Synthesizer
from waken_voice.transcribers import (
    GroqWhisperTranscriber,
    OpenAIWhisperTranscriber,
    Transcriber,
)
from waken_voice.wakeword import OpenWakeWordDetector, WakeWordDetector

__all__ = [
    "GTTSSynthesizer",
    "GroqWhisperTranscriber",
    "MicrophoneSource",
    "OpenAITTSSynthesizer",
    "OpenAIWhisperTranscriber",
    "OpenWakeWordDetector",
    "Synthesizer",
    "Transcriber",
    "VoiceOutput",
    "VoiceSource",
    "WakeWordDetector",
]
