import math
import time
from collections import defaultdict


class LatencyTracker:
    def __init__(self):
        self.durations = defaultdict(list)

    def record(self, stage: str, seconds: float) -> None:
        self.durations[stage].append(seconds)

    def _percentile(self, values: list[float], percentile: float) -> float:
        if not values:
            return 0.0

        ordered = sorted(values)

        if len(ordered) == 1:
            return ordered[0]

        position = (len(ordered) - 1) * (percentile / 100.0)

        lower_index = math.floor(position)
        upper_index = math.ceil(position)

        if lower_index == upper_index:
            return ordered[lower_index]

        lower_value = ordered[lower_index]
        upper_value = ordered[upper_index]

        fraction = position - lower_index

        return (
            lower_value
            + (upper_value - lower_value) * fraction
        )

    def summary(self) -> dict:
        output = {}

        for stage, values in self.durations.items():
            if not values:
                continue

            output[stage] = {
                "last": values[-1],
                "count": len(values),
                "total": sum(values),
                "min": min(values),
                "max": max(values),
                "p50": self._percentile(values, 50),
                "p95": self._percentile(values, 95),
            }

        return output

    def print_summary(self) -> None:
        summaries = self.summary()
        if not summaries:
            print("No latency measurements recorded.")
            return
        
        print()
        print("=" * 72)
        print("Relay latency summary")
        print("=" * 72)

        for stage, stats in summaries.items():
            print(
                f"{stage:15s} "
                f"count={stats['count']:3d}  "
                f"p50={stats['p50'] * 1000:7.0f} ms  "
                f"p95={stats['p95'] * 1000:7.0f} ms  "
                f"min={stats['min'] * 1000:7.0f} ms  "
                f"max={stats['max'] * 1000:7.0f} ms"
            )
        print("=" * 72)

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

        tracker.record(
            self.stage,
            elapsed,
        )
        print(
            f"[latency] {self.stage}: "
            f"{elapsed * 1000:.0f}ms"
        )