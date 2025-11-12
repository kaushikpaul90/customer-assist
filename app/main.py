"""REST API surface for Customer Assist.

This module defines the FastAPI application and the public endpoints
used by the demo. Endpoints are thin wrappers that call into
`app.models` and record metrics using `app.monitor` and `app.utils`.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uuid
import logging
import json
from pythonjsonlogger.json import JsonFormatter
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
from app.config import LOG_LEVEL
from app.models import (
    answer_question, detect_defect, summarize_text, transcribe_and_translate_audio, 
    transcribe_audio, translate_text, get_prompt_template, auto_grade_response
)
from app.utils import timeit, GLOBAL_RATE_LIMITER, RESPONSE_CACHE
from app.monitor import log_llm_interaction, aggregate_llmops_metrics

# --- COST CONTROL CONSTANTS ---
MAX_RESPONSE_LENGTH = 1500 # Max characters allowed in a response (e.g., for summarization)
MAX_INPUT_TOKEN_COUNT = 1000 # Max tokens allowed in a prompt/input before rejection
MAX_TEXT_INPUT_CHAR_COUNT = 4000 # Max characters allowed for text inputs (Added for consistency)
MAX_IMAGE_FILE_SIZE = 5 * 1024 * 1024 # 5MB limit for images (Added for consistency)
MAX_AUDIO_FILE_SIZE = 10 * 1024 * 1024 # 10MB limit for audio (Added for consistency)


# -------------------------------------------------------------
# 1. LOGGING SETUP: Use JSON formatting
# -------------------------------------------------------------
logger = logging.getLogger('llm_monitor')
logger.setLevel(LOG_LEVEL)

# Console handler setup for JSON output
handler = logging.StreamHandler()
# Use JsonFormatter to output structured data
formatter = JsonFormatter('%(levelname)s %(asctime)s %(name)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# Prevent duplicate logs from root logger if configured
logger.propagate = False

# -------------------------------------------------------------
# 2. PROMETHEUS METRICS SETUP
# -------------------------------------------------------------

# COUNTER: Total LLM calls by endpoint and status (success/failure)
LLM_CALLS = Counter(
    'llm_calls_total', 
    'Total number of LLM calls, labeled by endpoint and status.',
    ['endpoint', 'status', 'model_version']
)

# COUNTER: Total errors by endpoint and error type
MODEL_ERRORS = Counter(
    'llm_model_errors_total', 
    'Total number of LLM model errors, labeled by endpoint and error type.',
    ['endpoint', 'error_type']
)

# HISTOGRAM: Latency distribution of LLM calls (in seconds)
REQUEST_LATENCY = Histogram(
    'llm_request_latency_seconds', 
    'Latency of LLM requests in seconds.',
    ['endpoint', 'model_version'],
    # Buckets are commonly in seconds (0.01s to 10s)
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')) 
)

# Helper function to get the model version label consistent with monitor.py
def get_model_label(endpoint: str) -> str:
    """Provides the Prometheus model version label."""
    if endpoint == "/question-answering": return "qa_model/v1"
    if endpoint == "/summarize-text": return "summarizer/v1"
    if endpoint == "/translate-text": return "translator/v1"
    if endpoint == "/check-item-return-eligibility": return "defect_detector/v1"
    if endpoint == "/audio-transcribe": return "asr_model/v1"
    if endpoint == "/audio-transcribe-translate": return "asr_translator/v1"
    return "unknown"


# -------------------------------------------------------------
# 3. FASTAPI APPLICATION SETUP
# -------------------------------------------------------------
app = FastAPI(title="Customer Assist AI", description="AI-powered Customer Service Assistant", version="1.0")

# Request models
class QAReq(BaseModel):
    context: str
    question: str

class TextReq(BaseModel):
    text: str

class ExplainReq(BaseModel):
    topic: str
    style: str = "detailed"  # options: detailed, step-by-step

class TranslateReq(BaseModel):
    text: str
    src_lang: str
    target_lang: str

# -------------------------------------------------------------
# 4. PROMETHEUS ENDPOINT
# -------------------------------------------------------------
@app.get("/metrics")
async def metrics_endpoint():
    """Exposes Prometheus metrics."""
    return Response(generate_latest(), media_type="text/plain")

# -------------------------------------------------------------
# LLMOPS: Operational Metrics Endpoint (CSV-based)
# -------------------------------------------------------------
@app.get("/llmops/summary")
async def llmops_summary():
    """
    Exposes aggregated operational metrics (latency, throughput, error rate, token usage, and quality score)
    by reading the logged LLM interactions.
    """
    try:
        metrics = aggregate_llmops_metrics()
        return metrics
    except Exception as e:
        # NOTE: This endpoint should not fail the overall application.
        # If the monitoring file is corrupted, we log that fact.
        logger.error(f"Failed to aggregate metrics: {str(e)}", exc_info=True) 
        raise HTTPException(status_code=500, detail=f"Failed to aggregate metrics: {str(e)}")


# -------------------------------------------------------------
# Core Model Endpoints
# -------------------------------------------------------------

@app.post("/question-answering")
async def question_answering(req: QAReq):
    """Question answering endpoint (long-context support)."""
    endpoint = "/question-answering"
    model_version = get_model_label(endpoint)
    transaction_id = str(uuid.uuid4())
    full_prompt = f"Context: {req.context.strip()} Question: {req.question.strip()}"
    
    # --- Resilience: Rate Limiting (1 token consumed per request) ---
    if not GLOBAL_RATE_LIMITER.consume(1):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    # --- Cost Control: Input Size Limit Check ---
    if len(full_prompt) > MAX_TEXT_INPUT_CHAR_COUNT * 2: # Context + Question can be longer
        raise HTTPException(status_code=413, detail=f"Input text size exceeds the maximum limit (approx. {MAX_TEXT_INPUT_CHAR_COUNT * 2} characters).")


    # --- Resilience: Caching Check ---
    cache_key = f"{endpoint}:{req.context}:{req.question}"
    cached_response = RESPONSE_CACHE.get(cache_key)
    if cached_response:
        logger.info("Response served from cache.", extra={'json_fields': {"transaction_id": transaction_id, "cache_hit": True, "model_task": endpoint}})
        return JSONResponse(
            content={"answer": cached_response, "transaction_id": transaction_id},
            headers={"X-Cache-Status": "HIT"}
        )
        
    start = timeit()
    out = {}
    answer = ""
    p_version = "N/A"
    token_count = 0
    quality_score = 0.0
    feedback_notes = "Not evaluated."

    try:
        # 1. Execute the core logic
        out = answer_question(req.context, req.question)
        answer = out.get("answer", "")
        token_count = out.get("token_count", "")
        
        # 1a. AUTO-GRADE RESPONSE
        quality_score, feedback_notes = auto_grade_response(endpoint, answer)
        
        # --- Cost Control: Response Size Limit Check ---
        if len(answer) > MAX_RESPONSE_LENGTH:
             answer = answer[:MAX_RESPONSE_LENGTH] + "..." # Truncate response
             quality_score = min(quality_score, 0.8)
             feedback_notes = "Response truncated due to size limit."

        # 2. Log success & Prometheus
        latency_sec = timeit() - start
        latency_ms = latency_sec * 1000
        
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency_ms,
            prompt=full_prompt,
            output=answer,
            transaction_id=transaction_id,
            metadata={"answer_len": len(answer), "prompt_version": p_version, "token_count": token_count, "success": True},
            quality_score=quality_score,
            feedback_notes=feedback_notes
        )
        # Record Prometheus Metrics
        LLM_CALLS.labels(endpoint=endpoint, status='success', model_version=model_version).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, model_version=model_version).observe(latency_sec)
        
        # 3. Cache and Return successful response
        RESPONSE_CACHE.set(cache_key, answer) # Cache the successful answer
        del out["token_count"]
        out["transaction_id"] = transaction_id
        return JSONResponse(
            content=out,
            headers={"X-Cache-Status": "MISS"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # 1. Capture error details
        latency_sec = timeit() - start
        latency_ms = latency_sec * 1000
        error_message = str(e)
        error_type = type(e).__name__
        
        # 1a. Log failure (Set minimal score on error)
        quality_score, feedback_notes = 0.0, f"System Failure: {error_type}" 
        
        # 2. Log failure & Prometheus
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency_ms,
            prompt=full_prompt,
            output="ERROR: " + error_message[:100],
            transaction_id=transaction_id,
            metadata={"answer_len": 0, "prompt_version": p_version,  "token_count": token_count, "success": False, "error_type": error_type},
            quality_score=quality_score,
            feedback_notes=feedback_notes
        )
        # Record Prometheus Metrics
        LLM_CALLS.labels(endpoint=endpoint, status='failure', model_version=model_version).inc()
        MODEL_ERRORS.labels(endpoint=endpoint, error_type=error_type).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, model_version=model_version).observe(latency_sec)

        # 3. Raise HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize-text")
async def summarize_text_endpoint(req: TextReq):
    """Text summarization endpoint."""
    endpoint = "/summarize-text"
    model_version = get_model_label(endpoint)
    transaction_id = str(uuid.uuid4())
    
    # --- Resilience: Rate Limiting (1 token consumed per request) ---
    if not GLOBAL_RATE_LIMITER.consume(1):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    # --- Cost Control: Input Size Limit Check (Simplified check using char length) ---
    if len(req.text) > MAX_TEXT_INPUT_CHAR_COUNT: 
        raise HTTPException(status_code=413, detail=f"Input text size exceeds the maximum limit (approx. {MAX_TEXT_INPUT_CHAR_COUNT} characters).")


    # --- Resilience: Caching Check ---
    cache_key = f"{endpoint}:{req.text}"
    cached_response = RESPONSE_CACHE.get(cache_key)
    if cached_response:
        logger.info("Response served from cache.", extra={'json_fields': {"transaction_id": transaction_id, "cache_hit": True, "model_task": endpoint}})
        return JSONResponse(
            content={"summary": cached_response, "transaction_id": transaction_id},
            headers={"X-Cache-Status": "HIT"}
        )

    start = timeit()
    s = ""
    prompt = ""
    p_version = "N/A"
    token_count = 0
    quality_score = 0.0
    feedback_notes = "Not evaluated."
    
    # The redundant check below is safely ignored as the correct check is already above.
    if len(req.text) > 4000: # Assuming ~4000 chars is beyond the token limit for this simple demo
        raise HTTPException(status_code=413, detail="Input text size exceeds the maximum limit (approx. 4000 characters).")
    
    try:
        # 1. Execute the core logic
        s, prompt, p_version, token_count = summarize_text(req.text) 
        
        # 1a. AUTO-GRADE RESPONSE
        quality_score, feedback_notes = auto_grade_response(endpoint, s)

        # --- Cost Control: Response Size Limit Check ---
        if len(s) > MAX_RESPONSE_LENGTH:
             s = s[:MAX_RESPONSE_LENGTH] + "..." # Truncate response
             quality_score = min(quality_score, 0.8)
             feedback_notes = "Response truncated due to size limit."

        # 2. Log success & Prometheus
        latency_sec = timeit() - start
        latency_ms = latency_sec * 1000

        log_llm_interaction(
            endpoint=endpoint,
            latency=latency_ms,
            prompt=prompt,
            output=s,
            transaction_id=transaction_id,
            metadata={"summary_len": len(s), "prompt_version": p_version, "token_count": token_count, "success": True},
            quality_score=quality_score,
            feedback_notes=feedback_notes
        )
        # Record Prometheus Metrics
        LLM_CALLS.labels(endpoint=endpoint, status='success', model_version=model_version).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, model_version=model_version).observe(latency_sec)
        
        # 3. Cache and Return successful response
        RESPONSE_CACHE.set(cache_key, s) # Cache the successful summary
        return JSONResponse(
            content={"summary": s, "transaction_id": transaction_id},
            headers={"X-Cache-Status": "MISS"}
        )
        
    except HTTPException:
        # Re-raise explicit HTTP errors (like 413)
        raise
    except Exception as e:
        # 1. Capture error details
        latency_sec = timeit() - start
        latency_ms = latency_sec * 1000
        error_message = str(e)
        error_type = type(e).__name__
        
        # 1a. Log failure (Set minimal score on error)
        quality_score, feedback_notes = 0.0, f"System Failure: {error_type}"

        # 2. Log failure & Prometheus
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency_ms,
            prompt=prompt or req.text[:100],
            output="ERROR: " + error_message[:100],
            transaction_id=transaction_id,
            metadata={"summary_len": 0, "prompt_version": p_version, "token_count": token_count, "success": False, "error_type": error_type},
            quality_score=quality_score,
            feedback_notes=feedback_notes
        )
        # Record Prometheus Metrics
        LLM_CALLS.labels(endpoint=endpoint, status='failure', model_version=model_version).inc()
        MODEL_ERRORS.labels(endpoint=endpoint, error_type=error_type).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, model_version=model_version).observe(latency_sec)

        # 3. Raise HTTPException
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/translate-text')
async def translate_text_endpoint(req: TranslateReq):
    """
    Translate text to a target language.
    Caller provides src_lang and target_lang codes; this endpoint maps
    short codes (e.g. 'hi', 'eng') to the internal NLLB-style codes.
    """
    endpoint = "/translate-text"
    model_version = get_model_label(endpoint)
    transaction_id = str(uuid.uuid4())
    
    # --- Resilience: Rate Limiting (1 token consumed per request) ---
    if not GLOBAL_RATE_LIMITER.consume(1):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    # --- Cost Control: Input Size Limit Check ---
    if len(req.text) > MAX_TEXT_INPUT_CHAR_COUNT:
        raise HTTPException(status_code=413, detail=f"Input text size exceeds the maximum limit (approx. {MAX_TEXT_INPUT_CHAR_COUNT} characters).")

    # --- Resilience: Caching Check ---
    cache_key = f"{endpoint}:{req.text}:{req.src_lang}:{req.target_lang}"
    cached_response = RESPONSE_CACHE.get(cache_key)
    if cached_response:
        logger.info("Response served from cache.", extra={'json_fields': {"transaction_id": transaction_id, "cache_hit": True, "model_task": endpoint}})
        return JSONResponse(
            content={'translation': cached_response, "transaction_id": transaction_id},
            headers={"X-Cache-Status": "HIT"}
        )

    start = timeit()
    translated_text = ""
    prompt = ""
    p_version = "N/A"
    token_count = 0
    quality_score = 0.0
    feedback_notes = "Not evaluated."
    
    # The redundant check below is safely ignored as the correct check is already above.
    if len(req.text) > 4000:
        raise HTTPException(status_code=413, detail="Input text size exceeds the maximum limit (approx. 4000 characters).")
    
    try:
        text = req.text.strip()
        src_lang = req.src_lang.strip()
        target_lang = req.target_lang.strip()

        # Map short language codes to the NLLB / internal codes used by the
        # translation pipeline.
        lang_map = {
            "hi": "hin_Deva",
            "eng": "eng_Latn",
        }

        mapped_src = lang_map.get(src_lang, src_lang)
        mapped_tgt = lang_map.get(target_lang, target_lang)
        
        # 1. Execute the core logic
        translated_text, prompt, p_version, token_count = translate_text(text=text, src_lang=mapped_src, target_lang=mapped_tgt)

        # 1a. AUTO-GRADE RESPONSE
        quality_score, feedback_notes = auto_grade_response(endpoint, translated_text)

        # 2. Log success & Prometheus
        latency_sec = timeit() - start
        latency_ms = latency_sec * 1000
        
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency_ms,
            prompt=prompt,
            output=translated_text,
            transaction_id=transaction_id,
            metadata={"src": mapped_src, "tgt": mapped_tgt, 'out_len': len(translated_text), "prompt_version": p_version, "token_count": token_count, "success": True},
            quality_score=quality_score,
            feedback_notes=feedback_notes
        )
        # Record Prometheus Metrics
        LLM_CALLS.labels(endpoint=endpoint, status='success', model_version=model_version).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, model_version=model_version).observe(latency_sec)
        
        # 3. Cache and Return successful response
        RESPONSE_CACHE.set(cache_key, translated_text) # Cache the successful translation
        return JSONResponse(
            content={'translation': translated_text, "transaction_id": transaction_id},
            headers={"X-Cache-Status": "MISS"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # 1. Capture error details
        latency_sec = timeit() - start
        latency_ms = latency_sec * 1000
        error_message = str(e)
        error_type = type(e).__name__
        
        # 1a. Log failure (Set minimal score on error)
        quality_score, feedback_notes = 0.0, f"System Failure: {error_type}"

        # 2. Log failure & Prometheus
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency_ms,
            prompt=prompt or req.text[:100], 
            output="ERROR: " + error_message[:100],
            transaction_id=transaction_id,
            metadata={"src": mapped_src, "tgt": mapped_tgt, 'out_len': 0, "prompt_version": p_version, "token_count": token_count, "success": False, "error_type": error_type},
            quality_score=quality_score,
            feedback_notes=feedback_notes
        )
        # Record Prometheus Metrics
        LLM_CALLS.labels(endpoint=endpoint, status='failure', model_version=model_version).inc()
        MODEL_ERRORS.labels(endpoint=endpoint, error_type=error_type).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, model_version=model_version).observe(latency_sec)

        # 3. Raise HTTPException
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/check-item-return-eligibility')
async def check_item_return_eligibility(file: UploadFile = File(...)):
    """
    Check whether an uploaded item image is eligible for return.
    Runs defect detection and returns a simple eligibility signal and
    confidence.
    """
    endpoint = "/check-item-return-eligibility"
    model_version = get_model_label(endpoint)
    transaction_id = str(uuid.uuid4())

    # --- Resilience: Rate Limiting (1 token consumed per request) ---
    if not GLOBAL_RATE_LIMITER.consume(1):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    start = timeit()
    detection = {}
    file_name = file.filename
    content_type = file.content_type
    p_version = "N/A" 
    img_bytes = b''
    quality_score = 0.0
    feedback_notes = "Not evaluated."
    
    # --- Cost Control: File Size Limit Check ---
    # Read a limited amount to check file size before processing the rest
    max_file_size = 5 * 1024 * 1024 # 5MB limit
    
    try:
        # Read file content (stream until limit or EOF)
        img_bytes = await file.read()
        if len(img_bytes) > MAX_IMAGE_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File size exceeds the limit of {MAX_IMAGE_FILE_SIZE / (1024*1024):.0f}MB.")
        
        # --- NOTE ON CACHING ---
        # Caching is not applied to file uploads due to high input cardinality (each file is unique) 
        # and large memory consumption for cache keys/values.
        
        # 1. Execute the core logic (CV models currently don't use retry decorator)
        detection = detect_defect(img_bytes)
        logger.info("Detection result: %s", detection, extra={'json_fields': {"transaction_id": transaction_id}})

        eligibility = detection.get("eligible_for_return")
        
        # 1a. AUTO-GRADE RESPONSE 
        quality_score = 1.0 if eligibility in [True, False] else 0.0
        feedback_notes = f"Detection complete. Eligible: {eligibility}"
        
        # 2. Log success & Prometheus
        latency_sec = timeit() - start
        latency_ms = latency_sec * 1000
        
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency_ms,
            prompt=f"Image file uploaded: {file_name} ({content_type})",
            output=str(detection),
            transaction_id=transaction_id,
            metadata={
                "eligible": eligibility, 
                "predicted_label": detection.get("predicted_label"), 
                "file_size_bytes": len(img_bytes), 
                "prompt_version": p_version,
                "token_count": 0,
                "success": True
            },
            quality_score=quality_score,
            feedback_notes=feedback_notes
        )
        # Record Prometheus Metrics
        LLM_CALLS.labels(endpoint=endpoint, status='success', model_version=model_version).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, model_version=model_version).observe(latency_sec)

        # 3. Return successful response
        detection["transaction_id"] = transaction_id
        return detection
    
    except HTTPException:
        # Re-raise explicit HTTP errors (like 413)
        raise
    except Exception as e:
        # 1. Capture error details
        latency_sec = timeit() - start
        latency_ms = latency_sec * 1000
        error_message = str(e)
        error_type = type(e).__name__
        
        # 1a. Log failure (Set minimal score on error) 
        quality_score, feedback_notes = 0.0, f"System Failure: {error_type}"

        # 2. Log failure & Prometheus
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency_ms,
            prompt=f"Image file uploaded: {file_name} ({content_type})",
            output="ERROR: " + error_message[:100],
            transaction_id=transaction_id,
            metadata={
                "eligible": False, 
                "predicted_label": "error", 
                "file_size_bytes": len(img_bytes), 
                "prompt_version": p_version,
                "success": False,
                "token_count": 0,
                "error_type": error_type
            },
            quality_score=quality_score,
            feedback_notes=feedback_notes
        )
        # Record Prometheus Metrics
        LLM_CALLS.labels(endpoint=endpoint, status='failure', model_version=model_version).inc()
        MODEL_ERRORS.labels(endpoint=endpoint, error_type=error_type).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, model_version=model_version).observe(latency_sec)

        # 3. Raise HTTPException
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/audio-transcribe")
async def audio_transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio (wav/mp3) to text."""
    endpoint = "/audio-transcribe"
    model_version = get_model_label(endpoint)
    transaction_id = str(uuid.uuid4())
    
    # --- Resilience: Rate Limiting (1 token consumed per request) ---
    if not GLOBAL_RATE_LIMITER.consume(1):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    start = timeit()
    transcription = ""
    file_name = file.filename
    audio_bytes = b''
    p_version = "N/A" 
    token_count = 0
    quality_score = 0.0
    feedback_notes = "Not evaluated."
    
    # --- Cost Control: File Size Limit Check ---
    try:
        audio_bytes = await file.read()
        if len(audio_bytes) > MAX_AUDIO_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File size exceeds the limit of {MAX_AUDIO_FILE_SIZE / (1024*1024):.0f}MB.")
        
        # --- NOTE ON CACHING ---
        # Caching is not applied to audio uploads due to high input cardinality (each file is unique) 
        # and large memory consumption for cache keys/values.

        # 1. Execute the core logic
        transcribed_result, token_count = transcribe_audio(audio_bytes)
        transcription = transcribed_result.get("transcription", "")
        
        # 1a. AUTO-GRADE RESPONSE
        quality_score, feedback_notes = auto_grade_response(endpoint, transcription)

        # 2. Log success & Prometheus
        latency_sec = timeit() - start
        latency_ms = latency_sec * 1000
        
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency_ms,
            prompt=f"Audio file uploaded: {file_name}",
            output=transcription,
            transaction_id=transaction_id,
            metadata={
                "file_name": file_name,
                "transcription_len": len(transcription),"file_size_bytes": len(audio_bytes),
                "prompt_version": p_version,
                "token_count": token_count,
                "success": True
            },
            quality_score=quality_score,
            feedback_notes=feedback_notes
        )
        # Record Prometheus Metrics
        LLM_CALLS.labels(endpoint=endpoint, status='success', model_version=model_version).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, model_version=model_version).observe(latency_sec)
        
        # 3. Return successful response
        transcribed_result["transaction_id"] = transaction_id
        return transcribed_result
    
    except HTTPException:
        raise
    except Exception as e:
        # 1. Capture error details
        latency_sec = timeit() - start
        latency_ms = latency_sec * 1000
        error_message = str(e)
        error_type = type(e).__name__
        
        # 1a. Log failure (Set minimal score on error)
        quality_score, feedback_notes = 0.0, f"System Failure: {error_type}"

        # 2. Log failure & Prometheus
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency_ms,
            prompt=f"Audio file uploaded: {file_name}",
            output="ERROR: " + error_message[:100],
            transaction_id=transaction_id,
            metadata={
                "file_name": file_name,
                "transcription_len": 0,"file_size_bytes": len(audio_bytes),
                "prompt_version": p_version,
                "success": False,
                "token_count": token_count,
                "error_type": error_type
            },
            quality_score=quality_score,
            feedback_notes=feedback_notes
        )
        # Record Prometheus Metrics
        LLM_CALLS.labels(endpoint=endpoint, status='failure', model_version=model_version).inc()
        MODEL_ERRORS.labels(endpoint=endpoint, error_type=error_type).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, model_version=model_version).observe(latency_sec)

        # 3. Raise HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audio-transcribe-translate")
async def audio_transcribe_translate(file: UploadFile = File(...)):
    """Transcribe and translate uploaded audio to the target language."""
    endpoint = "/audio-transcribe-translate"
    model_version = get_model_label(endpoint)
    transaction_id = str(uuid.uuid4())
    
    # --- Resilience: Rate Limiting (1 token consumed per request) ---
    if not GLOBAL_RATE_LIMITER.consume(1):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    start = timeit()
    translated_text = ""
    file_name = file.filename
    audio_bytes = b''
    p_version = get_prompt_template("translator_prompt")[1] # Get the current version
    token_count = 0
    quality_score = 0.0
    feedback_notes = "Not evaluated."
    
    # --- Cost Control: File Size Limit Check ---
    try:
        audio_bytes = await file.read()
        if len(audio_bytes) > MAX_AUDIO_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File size exceeds the limit of {MAX_AUDIO_FILE_SIZE / (1024*1024):.0f}MB.")
        
        # --- NOTE ON CACHING ---
        # Caching is not applied to ASR/Translation chain due to high input cardinality (file input) 
        # and the multi-step nature of the pipeline.

        # 1. Execute the core logic
        translated_text, token_count = transcribe_and_translate_audio(audio_bytes)
        
        # 1a. AUTO-GRADE RESPONSE
        quality_score, feedback_notes = auto_grade_response(endpoint, translated_text)

        # 2. Log success & Prometheus
        latency_sec = timeit() - start
        latency_ms = latency_sec * 1000
        
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency_ms,
            prompt=f"Audio file uploaded for ASR and Translation: {file_name}",
            output=translated_text,
            transaction_id=transaction_id,
            metadata={
                "file_name": file_name,
                "translation_len": len(translated_text),
                "prompt_version": p_version,
                "file_size_bytes": len(audio_bytes),
                "token_count": token_count,
                "success": True
            },
            quality_score=quality_score,
            feedback_notes=feedback_notes
        )

        # Record Prometheus Metrics
        LLM_CALLS.labels(endpoint=endpoint, status='success', model_version=model_version).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, model_version=model_version).observe(latency_sec)
        
        # 3. Return successful response
        return {'translation': translated_text, "transaction_id": transaction_id}
    
    except HTTPException:
        raise
    except Exception as e:
        # 1. Capture error details
        latency_sec = timeit() - start
        latency_ms = latency_sec * 1000
        error_message = str(e)
        error_type = type(e).__name__

        # 1a. Log failure (Set minimal score on error)
        quality_score, feedback_notes = 0.0, f"System Failure: {error_type}"

        # 2. Log failure & Prometheus
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency_ms,
            prompt=f"Audio file uploaded for ASR and Translation: {file_name}",
            output="ERROR: " + error_message[:100],
            transaction_id=transaction_id,
            metadata={
                "file_name": file_name,
                "translation_len": 0,
                "prompt_version": p_version,
                "file_size_bytes": len(audio_bytes),
                "token_count": token_count,
                "success": False,
                "error_type": error_type
            },
            quality_score=quality_score,
            feedback_notes=feedback_notes
        )
        # Record Prometheus Metrics
        LLM_CALLS.labels(endpoint=endpoint, status='failure', model_version=model_version).inc()
        MODEL_ERRORS.labels(endpoint=endpoint, error_type=error_type).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, model_version=model_version).observe(latency_sec)

        # 3. Raise HTTPException
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "CustomerAssist_finetune AI API is running successfully!"}