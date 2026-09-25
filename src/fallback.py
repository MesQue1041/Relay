from dataclasses import dataclass
from queue import Queue
from threading import Thread
import time
from typing import Any, Callable

@dataclass
class StageResult:
    value: Any
    success: bool
    degraded: bool
    elapsed: float
    error: str | None = None

def run_with_timeout(
    function: Callable,
    timeout_seconds: float,
    fallback_value: Any,
    *args,
    **kwargs,
) -> StageResult:
    result_queue = Queue(maxsize=1)

    def worker():
        start = time.perf_counter()
        try:
            result = function(*args, **kwargs)
            elapsed = time.perf_counter() - start
            result_queue.put(
                StageResult(
                    value=result,
                    success=True,
                    degraded=False,
                    elapsed=elapsed,
                )
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            result_queue.put(
                StageResult(
                    value=fallback_value,
                    success=False,
                    degraded=True,
                    elapsed=elapsed,
                    error=str(exc),
                )
            )
    start = time.perf_counter()
    thread = Thread(
        target=worker,
        daemon=True,
    )
    thread.start()
    thread.join(timeout_seconds)
    elapsed = time.perf_counter() - start

    if thread.is_alive():
        return StageResult(
            value=fallback_value,
            success=False,
            degraded=True,
            elapsed=elapsed,
            error=(
                f"stage exceeded {timeout_seconds:.1f}s application budget"
            ),
        )
    try:
        return result_queue.get_nowait()
    except Exception:
        return StageResult(
            value=fallback_value,
            success=False,
            degraded=True,
            elapsed=elapsed,
            error="stage returned no result",
        )
LLM_FALLBACK_RESPONSE = (
    "I'm taking longer than expected to respond. "
    "Please try that again."
)

def run_llm_with_fallback(
    run_agent_function: Callable,
    user_text: str,
    timeout_seconds: float,
) -> StageResult:
    return run_with_timeout(
        run_agent_function,
        timeout_seconds,
        LLM_FALLBACK_RESPONSE,
        user_text,
    )

def run_tts_with_fallback(
    synthesize_function: Callable,
    text: str,
    timeout_seconds: float,
) -> StageResult:
    return run_with_timeout(
        synthesize_function,
        timeout_seconds,
        None,
        text,
    )