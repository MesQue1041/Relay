import os
import time

os.environ.setdefault("GROQ_API_KEY", "test-key")

from src.fallback import run_with_timeout
from src.guardrail import guardrail_response, validate_input
from src.latency import LatencyTracker

def test_guardrail_accepts_normal_input():
    result = validate_input("What is the capital of France?")
    assert result.allowed is True
    assert result.text == "What is the capital of France?"

def test_guardrail_rejects_empty_input():
    result = validate_input("")
    assert result.allowed is False

def test_guardrail_rejects_punctuation_only():
    result = validate_input("...")
    assert result.allowed is False

def test_guardrail_rejects_oversized_input():
    result = validate_input("hello " * 200)
    assert result.allowed is False
    assert guardrail_response(result.reason)

def test_latency_tracker_percentiles():
    tracker = LatencyTracker()
    for value in [0.1, 0.2, 0.3, 0.4, 0.5]:
        tracker.record("test", value)
    summary = tracker.summary()
    assert summary["test"]["count"] == 5
    assert summary["test"]["p50"] == 0.3
    assert summary["test"]["min"] == 0.1
    assert summary["test"]["max"] == 0.5

def test_fallback_success():
    def successful_function():
        return "success"
    result = run_with_timeout(successful_function, 1.0, "fallback")
    assert result.success is True
    assert result.degraded is False
    assert result.value == "success"

def test_fallback_timeout():
    def slow_function():
        time.sleep(0.5)
        return "finished"
    result = run_with_timeout(slow_function, 0.05, "fallback")
    assert result.success is False
    assert result.degraded is True
    assert result.value == "fallback"