import asyncio
import os
import tempfile

import edge_tts

from src.config import (
    TTS_VOICE,
    TTS_TIMEOUT_SECONDS,
)


async def _synthesize_async(text: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(
        text=text,
        voice=TTS_VOICE,
    )
    await asyncio.wait_for(
        communicate.save(output_path),
        timeout=TTS_TIMEOUT_SECONDS,
    )


def synthesize(text: str) -> str:
    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False,
    )

    output_path = temp_file.name
    temp_file.close()

    try:
        asyncio.run(
            _synthesize_async(
                text.strip(),
                output_path,
            )
        )

        if not os.path.exists(output_path):
            raise RuntimeError("TTS produced no output file.")

        if os.path.getsize(output_path) == 0:
            raise RuntimeError("TTS produced an empty audio file.")

        return output_path

    except asyncio.TimeoutError as exc:
        if os.path.exists(output_path):
            os.remove(output_path)

        raise RuntimeError(
            f"TTS timed out after {TTS_TIMEOUT_SECONDS:.1f}s."
        ) from exc

    except Exception as exc:
        if os.path.exists(output_path):
            os.remove(output_path)

        raise RuntimeError(
            f"TTS synthesis failed: {exc}"
        ) from exc