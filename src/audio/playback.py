import os
import time
from pathlib import Path

import pygame


_initialized = False


def _ensure_initialized() -> None:
    global _initialized

    if _initialized:
        return

    try:
        pygame.mixer.init()
        _initialized = True

    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize audio playback: {exc}"
        ) from exc


def play_audio(audio_path: str) -> None:
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    try:
        _ensure_initialized()

        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()

        # Wait until the response has finished playing.
        while pygame.mixer.music.get_busy():
            time.sleep(0.01)

    except Exception as exc:
        raise RuntimeError(
            f"Audio playback failed: {exc}"
        ) from exc


def cleanup_audio_file(audio_path: str) -> None:
    if not audio_path:
        return

    try:
        pygame.mixer.music.stop()
    except Exception:
        pass

    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    except OSError:
        pass


def shutdown() -> None:
    global _initialized

    if not _initialized:
        return

    try:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    finally:
        _initialized = False