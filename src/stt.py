import numpy as np
from faster_whisper import WhisperModel

from src.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)

print(
    f"[STT] Loading Whisper '{WHISPER_MODEL_SIZE}' model "
    f"({WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE})... this can take a few seconds.",
    flush=True,
)
_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)

_model.transcribe(np.zeros(16000, dtype=np.float32), language="en")
print("[STT] Whisper model ready.", flush=True)


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
        beam_size=5,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()

    return text