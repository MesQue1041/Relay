import time
from collections import defaultdict


class LatencyTracker:
    def __init__(self):
        self.durations = defaultdict(list)

    def record(self, stage: str, seconds: float):
        self.durations[stage].append(seconds)

    def summary(self):
        out = {}
        for stage, values in self.durations.items():
            out[stage] = {
                "last": values[-1],
                "count": len(values),
                "total": sum(values),
            }
        return out


tracker = LatencyTracker()


class Timer:
    def __init__(self, stage: str):
        self.stage = stage
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._start
        tracker.record(self.stage, elapsed)
        print(f"[latency] {self.stage}: {elapsed*1000:.0f}ms")
