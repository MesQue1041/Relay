import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Copy .env.example to .env and paste in "
        "your key from https://console.groq.com/keys"
    )

LLM_MODEL = "llama-3.1-8b-instant"
LLM_TIMEOUT_SECONDS = 6.0
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"


TTS_VOICE = "en-US-AriaNeural"
TTS_TIMEOUT_SECONDS = 5.0

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
VAD_FRAME_DURATION_MS = 30  # must be 10, 20, or 30

VAD_AGGRESSIVENESS = 2

VAD_SILENCE_FRAMES_THRESHOLD = 25

SAMPLE_AUDIO_DIR = "data/sample_audio"
EVAL_CASES_PATH = "evals/cases.yaml"