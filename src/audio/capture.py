import queue
import numpy as np
import sounddevice as sd
from src.config import AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, VAD_FRAME_DURATION_MS, MIC_GAIN

FRAME_SAMPLES = int(AUDIO_SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)

def _soft_limit(x: np.ndarray, ceiling: float = 32000.0) -> np.ndarray:
    return np.tanh(x / ceiling) * ceiling


class MicStream:
    def __init__(self):
        self._queue = queue.Queue()
        self._stream = None
        self._enabled = True

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[MicStream] audio status: {status}")

        if not self._enabled:
            return

        amplified = indata.astype(np.float32) * MIC_GAIN
        limited = _soft_limit(amplified)
        pcm16 = np.clip(limited, -32768, 32767).astype(np.int16)

        self._queue.put(pcm16.tobytes())

    def start(self):
        self._stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            latency="low",   
            callback=self._callback,
        )
        self._stream.start()
        self._enabled = True
        return self

    def pause(self):
        self._enabled = False
        self.clear_pending()

    def resume(self):
        self.clear_pending()
        self._enabled = True

    def clear_pending(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self):
        self._enabled = False

        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self.clear_pending()

    def frames(self):
        while self._stream is not None:
            try:
                yield self._queue.get(timeout=0.5)
            except queue.Empty:
                continue