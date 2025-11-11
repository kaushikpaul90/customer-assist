"""REST API surface for Customer Assist.

This module defines the FastAPI application and the public endpoints
used by the demo. Endpoints are thin wrappers that call into
`app.models` and record simple latency metrics using `app.utils`.

Keep this file light: avoid heavy imports or model initialization here
so FastAPI's import time remains predictable. Heavy model code
belongs in `app.models` where it can be refactored into lazy factories.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.models import (
    answer_question, detect_defect, summarize_text, transcribe_and_translate_audio, transcribe_audio, translate_text, get_prompt_template
)
from app.utils import timeit, record_metric
from app.monitor import log_llm_interaction, aggregate_llmops_metrics

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
# LLMOPS: Operational Metrics Endpoint
# -------------------------------------------------------------
@app.get("/llmops/summary")
async def llmops_summary():
    """
    Exposes aggregated operational metrics (latency, throughput, error rate, token usage)
    by reading the logged LLM interactions.
    """
    try:
        metrics = aggregate_llmops_metrics()
        return metrics
    except Exception as e:
        # NOTE: This endpoint should not fail the overall application.
        # If the monitoring file is corrupted, we log that fact.
        print(f"Error reading metrics file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to aggregate metrics: {str(e)}")


# -------------------------------------------------------------
# LLMOPS: Robust Error Logging Implementation
# -------------------------------------------------------------

@app.post("/question-answering")
async def question_answering(req: QAReq):
    """Question answering endpoint (long-context support)."""
    endpoint = "/question-answering"
    start = timeit()
    
    # Variables to track success/failure and output for logging
    out = {}
    p_version = "N/A"
    token_count = 0
    
    # Determine log content early
    full_prompt = f"Context: {req.context.strip()} | Question: {req.question.strip()}"

    try:
        # 1. Execute the core logic
        out = answer_question(req.context, req.question)
        answer = out.get("answer", "")
        token_count = out.get("token_count", "")
        
        # 2. Log success
        latency = (timeit() - start) * 1000
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=full_prompt,
            output=answer,
            metadata={"answer_len": len(answer), "prompt_version": p_version, "token_count": token_count, "success": True}
        )
        record_metric(endpoint, latency, {"answer_len": len(answer)})
        
        # 3. Return successful response
        del out["token_count"]
        return out
        
    except Exception as e:
        # 1. Capture error details
        latency = (timeit() - start) * 1000
        error_message = str(e)
        
        # 2. Log failure
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=full_prompt,
            output="ERROR: " + error_message[:100], # Log partial error message
            metadata={"answer_len": 0, "prompt_version": p_version,  "token_count": token_count, "success": False, "error_type": type(e).__name__}
        )
        record_metric(endpoint, latency, {"error": type(e).__name__})

        # 3. Raise HTTPException to return the error to the client
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/summarize-text")
async def summarize_text_endpoint(req: TextReq):
    """Text summarization endpoint."""
    endpoint = "/summarize-text"
    start = timeit()

    s = ""
    prompt = ""
    p_version = "N/A"
    token_count = 0

    try:
        # 1. Execute the core logic
        s, prompt, p_version, token_count = summarize_text(req.text) 
        
        # 2. Log success
        latency = (timeit() - start) * 1000
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=prompt,
            output=s,
            metadata={"summary_len": len(s), "prompt_version": p_version, "token_count": token_count, "success": True}
        )
        record_metric(endpoint, latency, {"summary_len": len(s)})
        
        # 3. Return successful response
        return {"summary": s}
        
    except Exception as e:
        # 1. Capture error details
        latency = (timeit() - start) * 1000
        error_message = str(e)
        
        # 2. Log failure
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=prompt or req.text[:100], # Log original text if prompt failed to generate
            output="ERROR: " + error_message[:100],
            metadata={"summary_len": 0, "prompt_version": p_version, "token_count": token_count, "success": False, "error_type": type(e).__name__}
        )
        record_metric(endpoint, latency, {"error": type(e).__name__})

        # 3. Raise HTTPException to return the error to the client
        raise HTTPException(status_code=500, detail=error_message)

@app.post('/translate-text')
async def translate_text_endpoint(req: TranslateReq):
    """
    Translate text to a target language.
    Caller provides src_lang and target_lang codes; this endpoint maps
    short codes (e.g. 'hi', 'eng') to the internal NLLB-style codes.
    """
    endpoint = "/translate-text"
    start = timeit()
    
    translated_text = ""
    prompt = ""
    p_version = "N/A"
    token_count = 0
    
    try:
        text = req.text.strip()
        src_lang = req.src_lang.strip()
        target_lang = req.target_lang.strip()

        # Map short language codes to the NLLB / internal codes used by the
        # translation pipeline. If a code is not recognized, pass it through
        # unchanged so callers can use other codes.
        lang_map = {
            "hi": "hin_Deva",
            "eng": "eng_Latn",
        }

        mapped_src = lang_map.get(src_lang, src_lang)
        mapped_tgt = lang_map.get(target_lang, target_lang)
        
        # 1. Execute the core logic
        translated_text, prompt, p_version, token_count = translate_text(text=text, src_lang=mapped_src, target_lang=mapped_tgt)

        # 2. Log success
        latency = (timeit() - start) * 1000
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=prompt,
            output=translated_text,
            metadata={"src": mapped_src, "tgt": mapped_tgt, 'out_len': len(translated_text), "prompt_version": p_version, "token_count": token_count, "success": True}
        )
        record_metric(endpoint, latency, {'out_len': len(translated_text)})
        
        # 3. Return successful response
        return {'translation': translated_text}
    
    except Exception as e:
        # 1. Capture error details
        latency = (timeit() - start) * 1000
        error_message = str(e)
        
        # 2. Log failure
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=prompt or req.text[:100], 
            output="ERROR: " + error_message[:100],
            metadata={"src": mapped_src, "tgt": mapped_tgt, 'out_len': 0, "prompt_version": p_version, "token_count": token_count, "success": False, "error_type": type(e).__name__}
        )
        record_metric(endpoint, latency, {"error": type(e).__name__})

        # 3. Raise HTTPException to return the error to the client
        raise HTTPException(status_code=500, detail=error_message)

@app.post('/check-item-return-eligibility')
async def check_item_return_eligibility(file: UploadFile = File(...)):
    """Check whether an uploaded item image is eligible for return.
    Runs defect detection and returns a simple eligibility signal and
    confidence.
    """
    endpoint = "/check-item-return-eligibility"
    start = timeit()
    
    detection = {}
    file_name = file.filename
    content_type = file.content_type
    p_version = "N/A" # Non-LLM endpoint
    
    try:
        img_bytes = await file.read()

        # 1. Execute the core logic
        detection = detect_defect(img_bytes)
        print("🩻 Detection result:", detection)
        
        eligibility = detection.get("eligible_for_return")
        
        # 2. Log success
        latency = (timeit() - start) * 1000
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=f"Image file uploaded: {file_name} ({content_type})",
            output=str(detection),
            metadata={
                "eligible": eligibility, 
                "predicted_label": detection.get("predicted_label"), 
                "file_size_bytes": len(img_bytes), 
                "prompt_version": p_version,
                "token_count": 0, # NEW: Non-LLM endpoint logs 0
                "success": True
            }
        )
        record_metric(endpoint, latency, {"eligible": eligibility, "predicted_label": detection.get("predicted_label")})

        # 3. Return successful response
        return detection
    
    except Exception as e:
        # 1. Capture error details
        latency = (timeit() - start) * 1000
        error_message = str(e)
        
        # 2. Log failure
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=f"Image file uploaded: {file_name} ({content_type})",
            output="ERROR: " + error_message[:100],
            metadata={
                "eligible": False, 
                "predicted_label": "error", 
                "file_size_bytes": 0, 
                "prompt_version": p_version,
                "success": False,
                "error_type": type(e).__name__
            }
        )
        record_metric(endpoint, latency, {"error": type(e).__name__})

        # 3. Raise HTTPException to return the error to the client
        raise HTTPException(status_code=500, detail=error_message)
    
@app.post("/audio-transcribe")
async def audio_transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio (wav/mp3) to text."""
    endpoint = "/audio-transcribe"
    start = timeit()
    
    transcription = ""
    file_name = file.filename
    audio_bytes = b''
    p_version = "N/A" # ASR is non-LLM text generation, no prompt versioning used
    token_count = 0
    
    try:
        audio_bytes = await file.read()
        
        # 1. Execute the core logic
        transcribed_result, token_count = transcribe_audio(audio_bytes)
        transcription = transcribed_result.get("transcription", "")
        
        # 2. Log success
        latency = (timeit() - start) * 1000
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=f"Audio file uploaded: {file_name}",
            output=transcription,
            metadata={
                "file_name": file_name,
                "transcription_len": len(transcription),"file_size_bytes": len(audio_bytes),
                "prompt_version": p_version,
                "token_count": token_count,
                "success": True
            }
        )
        record_metric(endpoint, latency, {"transcription_len": len(transcription)})
        
        # 3. Return successful response
        return transcribed_result
    
    except Exception as e:
        # 1. Capture error details
        latency = (timeit() - start) * 1000
        error_message = str(e)
        
        # 2. Log failure
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=f"Audio file uploaded: {file_name}",
            output="ERROR: " + error_message[:100],
            metadata={
                "file_name": file_name,
                "transcription_len": 0,"file_size_bytes": len(audio_bytes),
                "prompt_version": p_version,
                "success": False,
                "token_count": token_count,
                "error_type": type(e).__name__
            }
        )
        record_metric(endpoint, latency, {"error": type(e).__name__})

        # 3. Raise HTTPException to return the error to the client
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/audio-transcribe-translate")
async def audio_transcribe_translate(file: UploadFile = File(...)):
    """Transcribe and translate uploaded audio to the target language."""
    endpoint = "/audio-transcribe-translate"
    start = timeit()
    
    translated_text = ""
    file_name = file.filename
    audio_bytes = b''
    p_version = get_prompt_template("translator_prompt")[1] # Get the current version
    token_count = 0
    
    try:
        audio_bytes = await file.read()
        
        # 1. Execute the core logic
        translated_text, token_count = transcribe_and_translate_audio(audio_bytes)
        
        # 2. Log success
        latency = (timeit() - start) * 1000
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=f"Audio file uploaded for ASR and Translation: {file_name}",
            output=translated_text,
            metadata={
                "file_name": file_name,
                "translation_len": len(translated_text),
                "prompt_version": p_version,
                "file_size_bytes": len(audio_bytes),
                "token_count": token_count,
                "success": True
            }
        )

        record_metric(endpoint, latency, {"translation_len": len(translated_text)})
        
        # 3. Return successful response
        return {'translation': translated_text}
    
    except Exception as e:
        # 1. Capture error details
        latency = (timeit() - start) * 1000
        error_message = str(e)

        # 2. Log failure
        log_llm_interaction(
            endpoint=endpoint,
            latency=latency,
            prompt=f"Audio file uploaded for ASR and Translation: {file_name}",
            output="ERROR: " + error_message[:100],
            metadata={
                "file_name": file_name,
                "translation_len": 0,
                "prompt_version": p_version,
                "file_size_bytes": len(audio_bytes),
                "token_count": token_count,
                "success": False,
                "error_type": type(e).__name__
            }
        )
        record_metric(endpoint, latency, {"error": type(e).__name__})

        # 3. Raise HTTPException to return the error to the client
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/")
async def root():
    return {"message": "CustomerAssist_finetune AI API is running successfully!"}
