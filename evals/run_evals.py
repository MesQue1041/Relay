import sys
import time
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.agent import run_agent_with_trace

def load_cases():
    cases_path = PROJECT_ROOT / "evals" / "cases.yaml"

    with cases_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    return data["cases"]

def percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (
        (len(ordered) - 1)
        * percentile
        / 100
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = position - lower

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        ) * fraction
    )

def main():
    cases = load_cases()
    total = len(cases)
    routing_passes = 0
    response_passes = 0

    latencies = []
    
    print()
    print("=" * 72)
    print("Relay evaluation harness")
    print("=" * 72)

    for index, case in enumerate(cases, start=1):
        case_id = case["id"]
        utterance = case["utterance"]
        expected_tools = sorted(
            case.get(
                "expected_tools",
                [],
            )
        )
        budget = case.get(
            "latency_budget_seconds",
            10.0,
        )

        print()
        print(
            f"[{index}/{total}] "
            f"{case_id}"
        )
        print(
            f"Input: {utterance}"
        )
        start = time.perf_counter()

        try:
            result = run_agent_with_trace(
                utterance
            )
            elapsed = (
                time.perf_counter()
                - start
            )
            actual_tools = sorted(
                result.tools_used
            )
            routing_ok = (
                actual_tools
                == expected_tools
            )
            response_ok = bool(
                result.text
                and result.text.strip()
            )
            within_budget = (
                elapsed
                <= budget
            )
            if routing_ok:
                routing_passes += 1
            if response_ok:
                response_passes += 1
            latencies.append(
                elapsed
            )
            print(
                f"Tools expected: "
                f"{expected_tools}"
            )
            print(
                f"Tools actual:   "
                f"{actual_tools}"
            )
            print(
                f"Routing: "
                f"{'PASS' if routing_ok else 'FAIL'}"
            )
            print(
                f"Response: "
                f"{'PASS' if response_ok else 'FAIL'}"
            )
            print(
                f"Latency: "
                f"{elapsed * 1000:.0f}ms "
                f"({'PASS' if within_budget else 'OVER BUDGET'})"
            )
            print(
                f"Relay: {result.text}"
            )

        except Exception as exc:
            elapsed = (
                time.perf_counter()
                - start
            )
            latencies.append(
                elapsed
            )
            print(
                f"ERROR: {exc}"
            )

    p50 = percentile(
        latencies,
        50,
    )

    p95 = percentile(
        latencies,
        95,
    )

    routing_rate = (
        routing_passes
        / total
        * 100
        if total
        else 0
    )

    response_rate = (
        response_passes
        / total
        * 100
        if total
        else 0
    )
    print()
    print("=" * 72)
    print("Evaluation summary")
    print("=" * 72)
    print(
        f"Cases:            {total}"
    )
    print(
        f"Tool routing:     {routing_passes}/{total} "
        f"({routing_rate:.1f}%)"
    )
    print(
        f"Valid responses:  {response_passes}/{total} "
        f"({response_rate:.1f}%)"
    )
    print(
        f"Agent latency p50: {p50 * 1000:.0f}ms"
    )
    print(
        f"Agent latency p95: {p95 * 1000:.0f}ms"
    )
    print("=" * 72)

if __name__ == "__main__":
    main()