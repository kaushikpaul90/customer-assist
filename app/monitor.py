"""LLMOps Monitor: Logging LLM interactions for observability and quality analysis.

This module captures the inputs, outputs, and metadata for key LLM calls
to enable downstream monitoring and analysis (e.g., detecting prompt drift,
analyzing failure modes, or logging cost/latency).
"""

import csv
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

# Setup logging file
monitor_file = Path("metrics/llm_monitor.csv")
monitor_file.parent.mkdir(parents=True, exist_ok=True)
HEADER = ["timestamp", "model_task", "latency_ms", "prompt", "output", "metadata"]


def _get_active_model_version(task_name: str) -> str:
    """A placeholder to simulate looking up the active model version (LLMOps)."""
    # In a real system, this would query a proper Model Registry service.
    # Here, we use a simple mapping based on the task name.
    if task_name == "/question-answering":
        return "qa_model/v1"
    elif task_name == "/summarize-text":
        return "summarizer/v1"
    elif task_name == "/translate-text":
        return "translator/v1"
    # Add other tasks here as needed
    return task_name


def log_llm_interaction(
    endpoint: str,
    latency: float,
    prompt: str,
    output: str,
    metadata: dict = None,
):
    """
    Logs a single LLM interaction row to a CSV file for monitoring.

    Args:
        endpoint: The API endpoint that triggered the LLM (e.g., '/question-answering').
        latency: Latency in milliseconds.
        prompt: The full text prompt sent to the LLM.
        output: The full text response from the LLM.
        metadata: Optional dict for extra info (e.g., user ID, token count).
    """
    metadata = metadata or {}
    
    # Add model version to metadata for traceability
    metadata["model_version"] = _get_active_model_version(endpoint)

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_task": endpoint,
        "latency_ms": round(latency, 2),
        "prompt": prompt.replace('\n', ' ').strip(), # Single-line cleanup for CSV
        "output": output.replace('\n', ' ').strip(),
        "metadata": json.dumps(metadata)
    }

    new_file = not monitor_file.exists()
    with open(monitor_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

# -------------------------------------------------------------
# LLMOPS: Metric Aggregation
# -------------------------------------------------------------

def aggregate_llmops_metrics() -> dict:
    """
    Reads the llm_monitor.csv and calculates operational metrics.
    Metrics include total requests, error rate, latency percentiles (p50, p95),
    and throughput.
    """
    if not monitor_file.exists():
        return {
            "status": "No data yet.",
            "total_requests": 0,
            "overall_throughput_req_per_sec": 0.0,
            "per_endpoint_metrics": {}
        }

    # Data structure to hold metrics
    data = defaultdict(lambda: {'latencies': [], 'timestamps': [], 'errors': 0})
    all_timestamps = []
    total_requests = 0

    with open(monitor_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            endpoint = row['model_task']
            total_requests += 1
            
            # Error Detection: Use the logged metadata and output prefix
            is_error = False
            try:
                metadata = json.loads(row['metadata'])
                # Check for the explicit success flag
                if metadata.get('success') is False:
                    is_error = True
            except (json.JSONDecodeError, KeyError):
                # Fallback: Check if the output starts with "ERROR:"
                if row['output'].strip().startswith("ERROR:"):
                    is_error = True

            if is_error:
                data[endpoint]['errors'] += 1
            
            # Only include successful latencies in performance calculations
            # Successful requests will have a 'success': True flag, or simply not 'success': False
            if not is_error:
                try:
                    latency = float(row['latency_ms'])
                    data[endpoint]['latencies'].append(latency)
                except ValueError:
                    # Ignore non-numeric latency, which shouldn't happen for successful calls
                    pass

            # Timestamp processing for all requests (successful and failed)
            try:
                t = time.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
                epoch_time = time.mktime(t)
                data[endpoint]['timestamps'].append(epoch_time)
                all_timestamps.append(epoch_time)
            except ValueError:
                continue

    # 1. Overall Metrics Calculation
    if total_requests == 0:
        return {
            "status": "No complete data entries found.",
            "total_requests": 0,
            "overall_throughput_req_per_sec": 0.0,
            "per_endpoint_metrics": {}
        }
    
    min_time = min(all_timestamps)
    max_time = max(all_timestamps)
    time_window_sec = max_time - min_time
    if time_window_sec == 0:
        time_window_sec = 1 

    all_latencies = [l for endpoint_data in data.values() for l in endpoint_data['latencies']]
    total_errors = sum(d['errors'] for d in data.values())
    
    overall_metrics = {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": round(total_errors / total_requests, 4),
        "avg_latency_ms": round(np.mean(all_latencies), 2) if all_latencies else 0.0,
        "p50_latency_ms": round(np.percentile(all_latencies, 50), 2) if all_latencies else 0.0,
        "p95_latency_ms": round(np.percentile(all_latencies, 95), 2) if all_latencies else 0.0,
        "time_window_sec": round(time_window_sec, 2),
        "overall_throughput_req_per_sec": round(total_requests / time_window_sec, 2)
    }

    # 2. Per-Endpoint Metrics Calculation
    per_endpoint_metrics = {}
    for endpoint, d in data.items():
        requests = len(d['latencies']) + d['errors'] # Total requests = successful + errors
        errors = d['errors']
        
        endpoint_min_time = min(d['timestamps'])
        endpoint_max_time = max(d['timestamps'])
        endpoint_window_sec = endpoint_max_time - endpoint_min_time
        if endpoint_window_sec == 0:
            endpoint_window_sec = 1
        
        error_rate = round(errors / requests, 4) if requests > 0 else 0.0

        if d['latencies'] or requests > 0:
            per_endpoint_metrics[endpoint] = {
                "requests": requests,
                "errors": errors,
                "error_rate": error_rate,
                "avg_latency_ms": round(np.mean(d['latencies']), 2) if d['latencies'] else 0.0,
                "p50_latency_ms": round(np.percentile(d['latencies'], 50), 2) if d['latencies'] else 0.0,
                "p95_latency_ms": round(np.percentile(d['latencies'], 95), 2) if d['latencies'] else 0.0,
                "throughput_req_per_sec": round(requests / endpoint_window_sec, 2)
            }

    return {
        "overall_metrics": overall_metrics,
        "per_endpoint_metrics": per_endpoint_metrics
    }