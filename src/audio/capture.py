import queue
import sounddevice as sd
from src.config import AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, VAD_FRAME_DURATION_MS

FRAME_SAMPLES = int(AUDIO_SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)


class MicStream:
    def __init__(self):
        self._queue = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[MicStream] audio status: {status}")

        self._queue.put(indata.tobytes())

    def start(self):
        self._stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype="int16",             
            blocksize=FRAME_SAMPLES,   
                                      
            callback=self._callback,
        )
        self._stream.start()
        return self

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def frames(self):
        while self._stream is not None:
            try:
                yield self._queue.get(timeout=0.5)
            except queue.Empty:
                continue