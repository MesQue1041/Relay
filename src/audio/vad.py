from collections import deque

import numpy as np
import webrtcvad

from src.config import (
    AUDIO_SAMPLE_RATE,
    VAD_AGGRESSIVENESS,
    VAD_MIN_PEAK,
    VAD_PRE_ROLL_FRAMES,
    VAD_SILENCE_FRAMES_THRESHOLD,
    VAD_START_FRAMES,
)


class UtteranceDetector:
    def __init__(self):
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        self._buffer = bytearray()

        self._silence_run = 0
        self._speech_run = 0
        self._in_speech = False
        self._pre_roll = deque(maxlen=VAD_PRE_ROLL_FRAMES)

    def push(self, frame: bytes):
        audio = np.frombuffer(frame, dtype=np.int16)

        if audio.size == 0:
            return None

        peak = int(np.abs(audio).max())

        if peak < VAD_MIN_PEAK:
            is_speech = False
        else:
            is_speech = self._vad.is_speech(
                frame,
                AUDIO_SAMPLE_RATE,
            )

        if not self._in_speech:
            self._pre_roll.append(frame)

            if is_speech:
                self._speech_run += 1
            else:
                self._speech_run = 0

            if self._speech_run >= VAD_START_FRAMES:
                self._in_speech = True
                self._silence_run = 0
                self._buffer.extend(
                    b"".join(self._pre_roll)
                )
                self._pre_roll.clear()
                print("[VAD] Speech started")
            return None

        self._buffer.extend(frame)

        if is_speech:
            # Speech continues
            self._silence_run = 0
            return None

        # Silence detected
        self._silence_run += 1

        # User has been silent long enough to finish the utterance
        if self._silence_run >= VAD_SILENCE_FRAMES_THRESHOLD:
            utterance = bytes(self._buffer)

            print(
                "[VAD] Utterance complete "
                f"({len(utterance) / (AUDIO_SAMPLE_RATE * 2):.2f}s)"
            )

            self._reset()

            return utterance

        return None

    def _reset(self):
        self._buffer = bytearray()
        self._silence_run = 0
        self._speech_run = 0
        self._in_speech = False
        self._pre_roll.clear()