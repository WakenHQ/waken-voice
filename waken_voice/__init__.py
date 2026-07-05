"""Voice Source/Output for Waken.

`VoiceSource` turns audio files dropped on disk into `Event`s via a
pluggable `Transcriber` (OpenAI Whisper by default, or Groq's
Whisper-backed API); `VoiceOutput` turns a `Response`'s text into audio
files on disk via a pluggable `Synthesizer` (OpenAI TTS by default, or
`gTTS`), optionally playing them back through a platform audio player.

See the package README for the scope decision this is built around: a
file-drop channel, not a live microphone listener.
"""

from waken_voice.output import VoiceOutput
from waken_voice.source import VoiceSource
from waken_voice.synthesizers import GTTSSynthesizer, OpenAITTSSynthesizer, Synthesizer
from waken_voice.transcribers import (
    GroqWhisperTranscriber,
    OpenAIWhisperTranscriber,
    Transcriber,
)

__all__ = [
    "GTTSSynthesizer",
    "GroqWhisperTranscriber",
    "OpenAITTSSynthesizer",
    "OpenAIWhisperTranscriber",
    "Synthesizer",
    "Transcriber",
    "VoiceOutput",
    "VoiceSource",
]
