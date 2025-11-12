"""
Small utility helpers used across the demo application.
This module implements tiny helpers for measuring elapsed time,
retries with backoff, and rate limiting tools. (Updated for Resilience)
"""

import time
import random
import functools
from fastapi import HTTPException
from collections import deque
from datetime import datetime, timedelta

# --- Rate Limiting: Token Bucket Implementation ---
# Used to enforce API rate limits at the FastAPI level.
class RateLimiter:
    """Simple Token Bucket rate limiter."""
    def __init__(self, capacity: int, fill_rate: int, time_unit_seconds: int = 60):
        self.capacity = capacity  # Max tokens (burst size)
        self.fill_rate = fill_rate  # Tokens added per time unit
        self.time_unit_seconds = time_unit_seconds
        self.tokens = capacity
        self.last_update = time.time()

    def consume(self, num_tokens: int = 1) -> bool:
        """Attempts to consume tokens. Returns True if successful, False if rate limited."""
        now = time.time()
        time_elapsed = now - self.last_update
        
        # Refill tokens
        refill_amount = (time_elapsed / self.time_unit_seconds) * self.fill_rate
        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_update = now

        if self.tokens >= num_tokens:
            self.tokens -= num_tokens
            return True
        return False

# Global Rate Limiter instance (used in main.py)
# Example: 10 requests max burst, refilling at 5 tokens per 60 seconds.
GLOBAL_RATE_LIMITER = RateLimiter(capacity=10, fill_rate=5, time_unit_seconds=60)


# --- Caching: Simple In-Memory Cache ---
class Cache:
    """Simple in-memory cache with TTL (Time-To-Live)."""
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = timedelta(seconds=ttl_seconds)

    def get(self, key):
        """Retrieves item if not expired."""
        entry = self.cache.get(key)
        if entry:
            data, timestamp = entry
            if datetime.now() < timestamp + self.ttl:
                return data
            else:
                del self.cache[key] # Expire entry
        return None

    def set(self, key, value):
        """Sets item with current timestamp."""
        self.cache[key] = (value, datetime.now())

# Global Cache instance (used in main.py)
RESPONSE_CACHE = Cache(ttl_seconds=300) # Cache responses for 5 minutes


# --- Resilience: Retry with Exponential Backoff Decorator ---
def retry_with_backoff(
    max_retries=3, 
    initial_delay=1.0, 
    backoff_factor=2.0, 
    exception_to_catch=Exception
):
    """
    Decorator to retry a function call with exponential backoff and jitter.
    
    Args:
        max_retries: Maximum number of times to retry the function.
        initial_delay: Initial delay in seconds before the first retry.
        backoff_factor: Factor by which to multiply the delay (e.g., 2.0).
        exception_to_catch: The exception type(s) to catch and retry on.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exception_to_catch as e:
                    if attempt == max_retries:
                        # Re-raise the exception after the last attempt
                        raise e
                    
                    # Calculate delay with exponential backoff and jitter
                    jitter = random.uniform(0, 0.5 * delay) # Add up to 50% jitter
                    sleep_time = delay + jitter
                    
                    # Log retry attempt (we can't use the dedicated app logger here easily)
                    print(f"[{func.__name__}] Retrying in {sleep_time:.2f}s due to: {type(e).__name__}")
                    time.sleep(sleep_time)
                    delay *= backoff_factor
        return wrapper
    return decorator


def timeit():
    """Return current time in seconds since the epoch.

    Simple wrapper used by endpoints to compute latencies. This keeps
    calls consistent and easy to mock in tests.
    """
    return time.time()