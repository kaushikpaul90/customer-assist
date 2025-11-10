"""Small utility helpers used across the demo application.

This module implements tiny helpers for measuring elapsed time and
recording simple CSV metrics. The implementation is intentionally
minimal and stores metrics in `metrics/metrics.csv` under the project
root.
"""

import time
import csv
from pathlib import Path

# Ensure metrics folder exists and create file lazily when writing.
metrics_file = Path("metrics/metrics.csv")
metrics_file.parent.mkdir(parents=True, exist_ok=True)


def timeit():
    """Return current time in seconds since the epoch.

    Simple wrapper used by endpoints to compute latencies. This keeps
    calls consistent and easy to mock in tests.
    """
    return time.time()


def record_metric(endpoint, latency, metadata):
    """Save a single metric row to a CSV file.

    Args:
        endpoint: Logical name of the endpoint (e.g. '/qa').
        latency: Latency in milliseconds (float or int).
        metadata: Arbitrary metadata (will be stringified). Use a
            dict for structured values.
    """
    row = {
        "endpoint": endpoint,
        "latency_ms": round(latency, 2),
        "metadata": str(metadata),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    header = ["endpoint", "latency_ms", "metadata", "timestamp"]

    new_file = not metrics_file.exists()
    with open(metrics_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
