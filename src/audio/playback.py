import threading
import av
import numpy as np
import sounddevice as sd

OUTPUT_SAMPLE_RATE = 24000
OUTPUT_CHANNELS = 1

_stream_lock = threading.Lock()
_stream = None

def _get_stream():
    global _stream
    with _stream_lock:
        if _stream is None:
            _stream = sd.OutputStream(
                samplerate=OUTPUT_SAMPLE_RATE,
                channels=OUTPUT_CHANNELS,
                dtype="int16",
                blocksize=0,
                latency="low",
            )
            _stream.start()
    return _stream


def warmup():
    _get_stream()


def _write_frames(stream, resampler, frame):
    for resampled in resampler.resample(frame):
        data = resampled.to_ndarray()
        if data.ndim == 2:
            data = data.reshape(-1)
        data = np.asarray(data, dtype=np.int16)
        stream.write(data.reshape(-1, 1))


def play_audio_stream(audio_chunks, stop_event=None):
    stream = _get_stream()
    decoder = av.CodecContext.create("mp3", "r")
    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=OUTPUT_SAMPLE_RATE,
    )

    for audio_chunk in audio_chunks:
        if stop_event is not None and stop_event.is_set():
            return
        for packet in decoder.parse(audio_chunk):
            for frame in decoder.decode(packet):
                _write_frames(stream, resampler, frame)

    if stop_event is not None and stop_event.is_set():
        return

    for packet in decoder.parse(b""):
        for frame in decoder.decode(packet):
            _write_frames(stream, resampler, frame)

    for frame in decoder.decode(None):
        _write_frames(stream, resampler, frame)


def shutdown():
    global _stream
    with _stream_lock:
        if _stream is not None:
            try:
                _stream.stop()
                _stream.close()
            except Exception:
                pass
            _stream = None