import numpy as np
from faster_whisper import WhisperModel

from src.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)

_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)


def transcribe(pcm_bytes: bytes) -> str:
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0  # int16 range: -32768..32767

    segments, info = _model.transcribe(audio_float32, language="en")

    return " ".join(segment.text.strip() for segment in segments).strip()