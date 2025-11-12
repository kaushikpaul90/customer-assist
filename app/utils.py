"""
Small utility helpers used across the demo application.
This module implements tiny helpers for measuring elapsed time.
"""

import time
from pathlib import Path

def timeit():
    """Return current time in seconds since the epoch.

    Simple wrapper used by endpoints to compute latencies. This keeps
    calls consistent and easy to mock in tests.
    """
    return time.time()

