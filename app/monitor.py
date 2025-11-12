"""LLMOps Monitor: Logging LLM interactions for observability and quality analysis.

This module captures the inputs, outputs, and metadata for key LLM calls
to enable downstream monitoring and analysis (e.g., detecting prompt drift,
analyzing failure modes, or logging cost/latency, and model quality).
"""

import csv
import json
import time
import uuid 
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np

# Setup logger for structured logging
logger = logging.getLogger("llm_monitor")

# Setup logging file
monitor_file = Path("metrics/llm_monitor.csv")
monitor_file.parent.mkdir(parents=True, exist_ok=True)
HEADER = ["timestamp", "transaction_id", "model_task", "latency_ms", "prompt", "output", "metadata", "quality_score", "feedback_notes"]


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
    elif task_name == "/check-item-return-eligibility":
        return "defect_detector/v1"
    elif task_name == "/audio-transcribe":
        return "asr_model/v1"
    elif task_name == "/audio-transcribe-translate":
        # A combined task, we'll use a custom label
        return "asr_translator/v1"
    return task_name


def log_llm_interaction(
    endpoint: str,
    latency: float,
    prompt: str,
    output: str,
    metadata: dict = None,
    transaction_id: str = None,
    quality_score: float = None, 
    feedback_notes: str = ""
):
    """
    Logs a single LLM interaction row to a CSV file for monitoring AND
    logs a structured message to the Python logger.

    Args:
        endpoint: The API endpoint that triggered the LLM (e.g., '/question-answering').
        latency: Latency in milliseconds.
        prompt: The full text prompt sent to the LLM.
        output: The full text response from the LLM.
        metadata: Optional dict for extra info (e.g., user ID, token count).
        transaction_id: Unique ID for the interaction.
        quality_score: Automated quality score (0.0 to 1.0).
        feedback_notes: Automated feedback.
    """
    metadata = metadata or {}
    
    # Generate ID if not provided (should be provided by main.py)
    if transaction_id is None:
        # Fallback in case main.py didn't set it (but main.py should)
        transaction_id = str(uuid.uuid4())
    
    # Add model version to metadata for traceability
    model_version = _get_active_model_version(endpoint)
    metadata["model_version"] = model_version

    # -----------------------------------------------
    # 1. Structured Python Logger Output (JSON via python-json-logger in main.py)
    # -----------------------------------------------
    # Use the logger to output a structured message
    log_data = {
        "transaction_id": transaction_id, 
        "model_task": endpoint,
        "latency_ms": round(latency, 2),
        "quality_score": quality_score,
        "success": metadata.get("success", True),
        **metadata # Merge metadata fields directly (e.g., token_count, prompt_version)
    }
    
    # Log an INFO message with the structured data.
    # The actual JSON formatting is handled by the logger configuration in main.py.
    logger.info("LLM interaction completed", extra={'json_fields': log_data})
    
    # -----------------------------------------------
    # 2. CSV Monitor Output (Original implementation)
    # -----------------------------------------------
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_id": transaction_id, 
        "model_task": endpoint,
        "latency_ms": round(latency, 2),
        "prompt": prompt.replace('\n', ' ').strip(), 
        "output": output.replace('\n', ' ').strip(),
        "metadata": json.dumps(metadata),
        "quality_score": str(quality_score) if quality_score is not None else "", # Convert score to string
        "feedback_notes": feedback_notes.replace('\n', ' ').strip() # Sanitize notes
    }

    new_file = not monitor_file.exists()
    # Ensure fieldnames uses the full HEADER
    with open(monitor_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        
    return transaction_id


# -------------------------------------------------------------
# LLMOPS: Metric Aggregation
# -------------------------------------------------------------

def aggregate_llmops_metrics() -> dict:
    """
    Reads the llm_monitor.csv and calculates operationaland quality metrics.
    """
    if not monitor_file.exists():
        return {
            "status": "No data yet.",
            "total_requests": 0,
            "overall_throughput_req_per_sec": 0.0,
            "per_endpoint_metrics": {}
        }

    # Data structure to hold metrics
    data = defaultdict(lambda: {'latencies': [], 'timestamps': [], 'errors': 0, 'tokens': [], 'quality_scores': []})
    all_timestamps = []
    total_requests = 0
    all_quality_scores = [] 

    with open(monitor_file, 'r', newline='', encoding='utf-8') as f:
        # Use DictReader without specifying fieldnames; it will use the first row as the header
        reader = csv.DictReader(f) 
        
        # Determine if the 'transaction_id' and 'quality_score' columns exist (for backward compatibility)
        fieldnames = reader.fieldnames if reader.fieldnames else []
        has_transaction_id = 'transaction_id' in fieldnames
        has_quality_score = 'quality_score' in fieldnames

        for row in reader:
            endpoint = row.get('model_task')
            if not endpoint: continue # Skip malformed rows
                
            total_requests += 1
            
            # 1. Error Detection: Use the logged metadata and output prefix
            is_error = False
            metadata = {}
            try:
                metadata = json.loads(row.get('metadata', '{}'))
                # Check for the explicit success flag
                if metadata.get('success') is False:
                    is_error = True
            except (json.JSONDecodeError, KeyError):
                if row.get('output', '').strip().startswith("ERROR:"):
                    is_error = True

            if is_error:
                data[endpoint]['errors'] += 1
            
            # 2. Only include successful latencies in performance calculations
            # Successful requests will have a 'success': True flag, or simply not 'success': False
            if not is_error and row.get('latency_ms'):
                try:
                    # Latency
                    latency = float(row['latency_ms'])
                    data[endpoint]['latencies'].append(latency)
                    
                    # Token Count
                    token_count = metadata.get('token_count', 0)
                    if isinstance(token_count, (int, float)) and token_count >= 0:
                        data[endpoint]['tokens'].append(int(token_count))
                except (ValueError, json.JSONDecodeError, KeyError, TypeError):
                    # Ignore non-numeric latency, which shouldn't happen for successful calls
                    # Ignore corrupted rows for latency/token calculations
                    pass 
                    
            # 3. Quality Score Extraction (Safely access the column)
            if has_quality_score:
                quality_str = row.get('quality_score')
                if quality_str:
                    try:
                        score = float(quality_str)
                        data[endpoint]['quality_scores'].append(score)
                        all_quality_scores.append(score)
                    except ValueError:
                        pass 
                    
            # 4. Timestamp processing for all requests (successful and failed)
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
            "status": "No data yet.",
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
    all_tokens = [t for endpoint_data in data.values() for t in endpoint_data['tokens']]
    total_errors = sum(d['errors'] for d in data.values())
    
    overall_metrics = {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": round(total_errors / total_requests, 4),
        "avg_latency_ms": round(np.mean(all_latencies), 2) if all_latencies else 0.0,
        "p50_latency_ms": round(np.percentile(all_latencies, 50), 2) if all_latencies else 0.0,
        "p95_latency_ms": round(np.percentile(all_latencies, 95), 2) if all_latencies else 0.0,
        "total_tokens": sum(all_tokens), 
        "avg_tokens_per_request": round(np.mean(all_tokens), 2) if all_tokens else 0.0,
        "avg_quality_score": round(np.mean(all_quality_scores), 4) if all_quality_scores else 0.0, 
        "time_window_sec": round(time_window_sec, 2),
        "overall_throughput_req_per_sec": round(total_requests / time_window_sec, 2)
    }

    # 2. Per-Endpoint Metrics Calculation
    per_endpoint_metrics = {}
    for endpoint, d in data.items():
        requests = len(d['latencies']) + d['errors'] # Total requests = successful + errors
        errors = d['errors']
        
        # Calculate endpoint-specific time window
        endpoint_min_time = min(d['timestamps'])
        endpoint_max_time = max(d['timestamps'])
        endpoint_window_sec = endpoint_max_time - endpoint_min_time
        if endpoint_window_sec == 0:
            endpoint_window_sec = 1
        
        error_rate = round(errors / requests, 4) if requests > 0 else 0.0

        if d['latencies'] or requests > 0:
            tokens = d['tokens']
            quality_scores = d['quality_scores']

            per_endpoint_metrics[endpoint] = {
                "requests": requests,
                "errors": errors,
                "error_rate": error_rate,
                "avg_latency_ms": round(np.mean(d['latencies']), 2) if d['latencies'] else 0.0,
                "p50_latency_ms": round(np.percentile(d['latencies'], 50), 2) if d['latencies'] else 0.0,
                "p95_latency_ms": round(np.percentile(d['latencies'], 95), 2) if d['latencies'] else 0.0,
                "total_tokens": sum(tokens), 
                "avg_tokens_per_request": round(np.mean(tokens), 2) if tokens else 0.0, 
                "avg_quality_score": round(np.mean(quality_scores), 4) if quality_scores else 0.0, 
                "throughput_req_per_sec": round(requests / endpoint_window_sec, 2)
            }

    return {
        "overall_metrics": overall_metrics,
        "per_endpoint_metrics": per_endpoint_metrics
    }