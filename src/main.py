from src.audio.capture import MicStream
from src.audio.vad import UtteranceDetector
from src.audio.playback import (
    play_audio,
    cleanup_audio_file,
    shutdown,
)

from src.stt import transcribe
from src.agent import run_agent
from src.latency import Timer
from src.tts import synthesize


def main():
    mic = MicStream().start()
    detector = UtteranceDetector()

    print()
    print("=" * 60)
    print("Relay")
    print("=" * 60)
    print("Listening...")
    print("Speak normally, then pause for about 1 second.")
    print("Relay will answer out loud.")
    print("Press Ctrl+C to quit.")
    print("=" * 60)
    print()

    try:
        for frame in mic.frames():
            # Stage 1: VAD
            utterance = detector.push(frame)

            if utterance is None:
                continue

            # Stage 2: STT
            print("[STT] Transcribing...")

            with Timer("stt"):
                text = transcribe(utterance)

            if not text:
                print(
                    "(heard something, but couldn't transcribe "
                    "any words)"
                )
                print()
                print("Listening...")
                continue

            print(f"You said: {text}")

            # Stage 3: LLM + tools
            print("[LLM] Thinking...")

            try:
                with Timer("llm"):
                    reply = run_agent(text)

            except Exception as exc:
                print(f"[LLM] Error: {exc}")
                print()
                print("Listening...")
                continue

            if not reply:
                print("Relay: (empty response)")
                print()
                print("Listening...")
                continue

            print(f"Relay: {reply}")

            # Stage 4: TTS
            audio_path = None

            try:
                print("[TTS] Synthesizing...")

                with Timer("tts"):
                    audio_path = synthesize(reply)

                print("[TTS] Playing response...")

                with Timer("playback"):
                    play_audio(audio_path)

            except Exception as exc:
                # Graceful degradation
                print(f"[TTS] Audio response unavailable: {exc}")

            finally:
                if audio_path:
                    cleanup_audio_file(audio_path)

            print()
            print("Listening...")

    except KeyboardInterrupt:
        print()
        print("Stopping.")

    finally:
        mic.stop()
        shutdown()


if __name__ == "__main__":
    main()