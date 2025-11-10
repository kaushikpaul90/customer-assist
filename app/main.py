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
    answer_question, detect_defect, summarize_text, explain_topic,
    ocr_image_bytes, classify_image_bytes, transcribe_and_translate_audio, transcribe_audio,
    translate_text
)
from app.utils import timeit, record_metric

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

@app.post("/qa")
async def qa(req: QAReq):
    """Question Answering Endpoint"""
    try:
        start = timeit()
        out = answer_question(req.context, req.question)
        latency = (timeit() - start) * 1000
        record_metric("/qa", latency, {"answer_len": len(out.get("answer", ""))})
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize(req: TextReq):
    """Text Summarization Endpoint"""
    try:
        start = timeit()
        s = summarize_text(req.text)
        latency = (timeit() - start) * 1000
        record_metric("/summarize", latency, {"summary_len": len(s)})
        return {"summary": s}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain(req: ExplainReq):
    """Topic Explanation Endpoint"""
    try:
        start = timeit()
        out = explain_topic(req.topic, style=req.style)
        latency = (timeit() - start) * 1000
        record_metric("/explain", latency, {"out_len": len(out)})
        
        return {'explanation': out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """OCR + Image Classification Endpoint"""
    try:
        start = timeit()
        img_bytes = await file.read()
        ocr = ocr_image_bytes(img_bytes)
        s = summarize_text(ocr["text"])
        labels = classify_image_bytes(img_bytes)
        latency = (timeit() - start) * 1000
        record_metric("/upload-image", latency, {"ocr_len": len(ocr)})
        # return {"ocr": ocr, "labels": labels}
        return {"summary": s, "labels": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/translate')
async def translate(req: TranslateReq):
    """
    Translate text to a target language.
    Target language is specified as a suffix in the input text, e.g., "Hello world. [fr]"
    """
    start = timeit()
    try:
        text = req.text.strip()
        src_lang = req.src_lang.strip()
        target_lang = req.target_lang.strip()
        translated_text = translate_text(text=text, src_lang=src_lang, target_lang=target_lang)
        latency = (timeit()-start)*1000
        record_metric('/translate', latency, {'out_len': len(translated_text)})
        return {'translation': translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/scan-item')
async def caption(file: UploadFile = File(...)):
    """
    Generate a short caption for an uploaded image.
    """
    start = timeit()
    try:
        img_bytes = await file.read()

        # Step 1: Detect defects
        detection = detect_defect(img_bytes)
        print("🩻 Detection result:", detection)

        return {
            "detection": detection,
            # "explanation": description if detection["is_defective"] else "No defects detected."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/asr")
async def asr(file: UploadFile = File(...)):
    """Transcribe uploaded audio (wav/mp3) to text."""
    audio_bytes = await file.read()
    return transcribe_audio(audio_bytes)
    
@app.post("/asr-translate")
async def asr(file: UploadFile = File(...)):
    """Transcribe and translate uploaded audio (wav/mp3) to English text."""
    audio_bytes = await file.read()
    translated_text = transcribe_and_translate_audio(audio_bytes)
    return {'translation': translated_text}

@app.get("/")
async def root():
    return {"message": "CustomerAssisst_finetune AI API is running successfully!"}
