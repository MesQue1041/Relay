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
    if not pcm_bytes:
        return ""

    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

    if audio_int16.size == 0:
        return ""

    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    segments, info = _model.transcribe(
        audio_float32,
        language="en",
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0.0,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()

    return text
