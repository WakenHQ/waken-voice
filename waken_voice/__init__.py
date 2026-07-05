"""Voice Source/Output for Waken.

`VoiceSource` turns audio files dropped on disk into `Event`s (speech-to-text
via OpenAI's Whisper API); `VoiceOutput` turns a `Response`'s text into audio
files on disk (text-to-speech via OpenAI's TTS API), optionally playing them
back through a platform audio player.

See the package README for the scope decision this is built around: a
file-drop channel, not a live microphone listener.
"""

from waken_voice.output import VoiceOutput
from waken_voice.source import VoiceSource

__all__ = ["VoiceOutput", "VoiceSource"]
