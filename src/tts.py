import edge_tts
from src.config import (
    TTS_VOICE,
    TTS_WORDS_PER_SECOND,
    TTS_CONNECT_OVERHEAD_SECONDS,
    TTS_MIN_TIMEOUT_SECONDS,
)

def stream_tts(text: str):
    if not text or not text.strip():
        return
    communicate = edge_tts.Communicate(
        text=text.strip(),
        voice=TTS_VOICE,
    )
    for chunk in communicate.stream_sync():
        if chunk["type"] == "audio" and chunk["data"]:
            yield chunk["data"]

def estimate_timeout_seconds(text: str) -> float:
    word_count = max(1, len(text.split()))
    estimated_speech_seconds = word_count / TTS_WORDS_PER_SECOND
    return max(
        TTS_MIN_TIMEOUT_SECONDS,
        estimated_speech_seconds + TTS_CONNECT_OVERHEAD_SECONDS,
    )