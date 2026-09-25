from collections import deque
import numpy as np
import webrtcvad

from src.config import (
    AUDIO_SAMPLE_RATE,
    VAD_AGGRESSIVENESS,
    VAD_MIN_PEAK,
    VAD_MIN_RMS,
    VAD_NOISE_MULTIPLIER,
    VAD_PRE_ROLL_FRAMES,
    VAD_SILENCE_FRAMES_THRESHOLD,
    VAD_START_FRAMES,
)

class UtteranceDetector:
    def __init__(self):
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._buffer = bytearray()
        self._pre_roll = deque(maxlen=VAD_PRE_ROLL_FRAMES)
        self._silence_run = 0
        self._speech_run = 0
        self._in_speech = False
        self._noise_rms = None

    def push(self, frame: bytes):
        audio = np.frombuffer(frame, dtype=np.int16)

        if audio.size == 0:
            return None

        peak = float(np.abs(audio).max())
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))

        web_rtc_speech = self._vad.is_speech(
            frame,
            AUDIO_SAMPLE_RATE,
        )
        energy_threshold = max(
            VAD_MIN_RMS,
            (self._noise_rms or VAD_MIN_RMS) * VAD_NOISE_MULTIPLIER,
        )
        energy_speech = (
            peak >= VAD_MIN_PEAK
            and rms >= energy_threshold
        )

        is_speech = web_rtc_speech and energy_speech
        if not self._in_speech:
            self._pre_roll.append(frame)

            if self._noise_rms is None:
                self._noise_rms = rms
            elif not web_rtc_speech:
                self._noise_rms = (
                    self._noise_rms * 0.95
                    + rms * 0.05
                )

            if is_speech:
                self._speech_run += 1
            else:
                self._speech_run = 0

            if self._speech_run >= VAD_START_FRAMES:
                self._in_speech = True
                self._silence_run = 0
                self._buffer.extend(b"".join(self._pre_roll))
                self._pre_roll.clear()
                print("[VAD] Speech started")
            return None
        self._buffer.extend(frame)

        if is_speech:
            self._silence_run = 0
            return None
        
        self._silence_run += 1
        if self._silence_run >= VAD_SILENCE_FRAMES_THRESHOLD:
            utterance = bytes(self._buffer)
            duration = len(utterance) / (
                AUDIO_SAMPLE_RATE * 2
            )
            print(
                f"[VAD] Utterance complete ({duration:.2f}s)"
            )
            self._reset()
            return utterance
        return None

    def _reset(self):
        self._buffer = bytearray()
        self._pre_roll.clear()
        self._silence_run = 0
        self._speech_run = 0
        self._in_speech = False