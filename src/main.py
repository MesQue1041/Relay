import os
import re
import signal
import time
from queue import Queue
from threading import Thread
from src.audio.capture import MicStream
from src.audio.vad import UtteranceDetector
from src.audio.playback import play_audio_stream, shutdown, warmup as tts_warmup
from src.agent import stream_agent
from src.config import LLM_BUDGET_SECONDS
from src.control import stop_event
from src.fallback import run_with_timeout
from src.guardrail import guardrail_response, validate_input
from src.latency import Timer, tracker
from src.stt import transcribe
from src.tts import stream_tts, estimate_timeout_seconds

def _handle_sigint(signum, frame):
    if stop_event.is_set():
        print("\n[Relay] Forcing exit.")
        os._exit(1)
    print("\n[Relay] Stopping... (press Ctrl+C again to force quit)")
    stop_event.set()


signal.signal(signal.SIGINT, _handle_sigint)

def extract_sentences(buffer: str):
    sentences = []
    while True:
        match = re.search(r"(.+?[.!?])(?:\s+|$)", buffer, re.S)
        if not match:
            break
        sentence = match.group(1).strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[match.end():]
    return buffer, sentences

def _prime_mic(mic, seconds: float = 0.3):
    deadline = time.perf_counter() + seconds
    for _ in mic.frames():
        if stop_event.is_set() or time.perf_counter() > deadline:
            break

def _wait_for_queue(audio_queue: Queue):
    while audio_queue.unfinished_tasks > 0:
        if stop_event.is_set():
            return
        time.sleep(0.05)

def _synthesize_and_play(sentence: str):
    start = time.perf_counter()
    first_audio = True

    def audio_generator():
        nonlocal first_audio
        for chunk in stream_tts(sentence):
            if first_audio:
                tracker.record("tts_ttfa", time.perf_counter() - start)
                first_audio = False
            yield chunk

    play_audio_stream(audio_generator(), stop_event=stop_event)
    tracker.record("tts", time.perf_counter() - start)
    return True

def tts_worker(audio_queue: Queue):
    while True:
        sentence = audio_queue.get()
        if sentence is None:
            audio_queue.task_done()
            break

        if stop_event.is_set():
            audio_queue.task_done()
            continue

        print(f"\n[TTS] Streaming: {sentence}")

        timeout = estimate_timeout_seconds(sentence)
        result = run_with_timeout(_synthesize_and_play, timeout, None, sentence)

        if not result.success:
            print(f"[Fallback] TTS unavailable/slow ({result.error}). Relay text: {sentence}")

        audio_queue.task_done()

def run_tts(reply: str):
    audio_queue = Queue()
    worker = Thread(target=tts_worker, args=(audio_queue,), daemon=True)
    worker.start()
    audio_queue.put(reply)
    audio_queue.put(None)
    _wait_for_queue(audio_queue)
    worker.join(timeout=1.0)

def main():
    mic = MicStream().start()
    print("[Mic] Warming up input stream...")
    _prime_mic(mic)

    print("[TTS] Warming up output stream...")
    tts_warmup()

    detector = UtteranceDetector()

    print()
    print("=" * 60)
    print("Relay")
    print("=" * 60)
    print("Listening...")
    print("Speak normally, then pause for about 1 second.")
    print("Relay will answer out loud.")
    print("Press Ctrl+C to quit (press twice to force quit immediately).")
    print("=" * 60)
    print()

    try:
        for frame in mic.frames():
            if stop_event.is_set():
                break

            utterance = detector.push(frame)
            if utterance is None:
                continue
            mic.pause()

            try:
                with Timer("turn_total"):
                    print("[STT] Transcribing...")
                    with Timer("stt"):
                        text = transcribe(utterance)

                    if not text:
                        print("(heard something, but couldn't transcribe any words)")
                        continue

                    print(f"You said: {text}")

                    guardrail = validate_input(text)
                    if not guardrail.allowed:
                        print(f"[Guardrail] Rejected: {guardrail.reason}")
                        reply = guardrail_response(guardrail.reason)
                        print(f"Relay: {reply}")
                        run_tts(reply)
                        continue

                    print("[LLM] Streaming...")

                    audio_queue = Queue()
                    worker = Thread(target=tts_worker, args=(audio_queue,), daemon=True)
                    worker.start()

                    print("Relay: ", end="", flush=True)

                    response_buffer = ""
                    first_token = True
                    llm_start = time.perf_counter()
                    deadline = llm_start + LLM_BUDGET_SECONDS

                    try:
                        for chunk in stream_agent(guardrail.text):
                            if stop_event.is_set():
                                break
                            if not chunk:
                                continue
                            if first_token:
                                tracker.record("llm_ttft", time.perf_counter() - llm_start)
                                first_token = False
                            print(chunk, end="", flush=True)
                            response_buffer += chunk
                            response_buffer, sentences = extract_sentences(response_buffer)
                            for sentence in sentences:
                                audio_queue.put(sentence)
                            if time.perf_counter() > deadline:
                                print("\n[Fallback] LLM exceeded its latency budget, cutting off here.")
                                break
                        if response_buffer.strip() and not stop_event.is_set():
                            audio_queue.put(response_buffer.strip())
                        print()
                    except Exception as exc:
                        print()
                        print(f"[Fallback] LLM failed: {exc}")
                        audio_queue.put("I ran into a problem answering that. Please try again.")
                    finally:
                        audio_queue.put(None)
                        _wait_for_queue(audio_queue)
                        worker.join(timeout=1.0)
            finally:
                mic.resume()
                detector = UtteranceDetector()

            if not stop_event.is_set():
                print()
                print("Listening...")

    except KeyboardInterrupt:
        pass  

    finally:
        print("\nStopping.")
        mic.stop()
        shutdown()
        tracker.print_summary()


if __name__ == "__main__":
    main()