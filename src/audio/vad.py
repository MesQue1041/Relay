import webrtcvad

from src.config import (
    AUDIO_SAMPLE_RATE,
    VAD_AGGRESSIVENESS,
    VAD_SILENCE_FRAMES_THRESHOLD,
)


class UtteranceDetector:
    def __init__(self):
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._buffer = bytearray()   # accumulated speech audio
        self._silence_run = 0        # consecutive silent frames counted
        self._in_speech = False      # has speech started this utterance?

    def push(self, frame: bytes):
        is_speech = self._vad.is_speech(frame, AUDIO_SAMPLE_RATE)

        if is_speech:
            self._in_speech = True
            self._silence_run = 0
            self._buffer.extend(frame)
            return None

        # This frame is silence.
        if not self._in_speech:
            return None

        self._silence_run += 1
        self._buffer.extend(frame)  

        if self._silence_run >= VAD_SILENCE_FRAMES_THRESHOLD:
            utterance = bytes(self._buffer)
            self._reset()
            return utterance

        return None

    def _reset(self):
        self._buffer = bytearray()
        self._silence_run = 0
        self._in_speech = False