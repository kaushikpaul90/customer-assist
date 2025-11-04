from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.models import (
    answer_question, detect_defect, explain_defect, infer_damage_from_caption, summarize_text, explain_topic,
    ocr_image_bytes, classify_image_bytes, detect_damage, transcribe_and_translate_audio, transcribe_audio,
    translate_text, image_caption_bytes
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

# class ParaphraseReq(BaseModel):
#     text: str
#     mode: str = 'paraphrase'  # or 'simplify'

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
    
        # # Format output into a proper numbered list if step-by-step
        # if req.style.lower() == "step-by-step":
        #     # Format for readable multi-line display
        #     formatted = out.replace("\\n", "\n") \
        #                 .replace("Step ", "\nStep ") \
        #                 .replace("step ", "\nStep ") \
        #                 .replace(".", ".\n")
        #     return formatted.strip()
        
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

# @app.post('/paraphrase')
# async def paraphrase(req: ParaphraseReq):
#     """
#     Paraphrase or simplify text.
#     mode: 'paraphrase' or 'simplify'
#     """
#     start = timeit()
#     try:
#         out = paraphrase_text(req.text, mode=req.mode)
#         latency = (timeit()-start)*1000
#         record_metric('/paraphrase', latency, {'out_len': len(out), 'mode': req.mode})
#         return {'result': out}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

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
        
        # # Step 1: Captioning
        # caption_result = image_caption_bytes(img_bytes)
        
        # # Step 2: Damage detection
        # # damage_result = detect_damage(img_bytes)
        # is_damaged_from_caption = infer_damage_from_caption(caption_result["caption"])

        # # Step 3: Get object context (optional for explainability)
        # # is_damaged_from_object_detection = detect_damage(img_bytes)
        # is_damaged_from_object_detection = classify_image_bytes(img_bytes)

        # # Step 4: Decide eligibility
        # is_damaged = is_damaged_from_caption or is_damaged_from_object_detection
        # eligibility = "Eligible for return" if is_damaged else "Not eligible for return"
        # latency = (timeit()-start)*1000
        # record_metric('/caption', latency, {'caption_len': len(caption_result.get('caption',''))})

        # return {
        #     # "item": cap["caption"],
        #     "eligibility": eligibility,
        #     # "latency_ms": round(latency, 2)
        # }

        # Step 1: Detect defects
        detection = detect_defect(img_bytes)
        print("🩻 Detection result:", detection)

        # # Step 2 (optional): Generate explanation if defective
        # if detection["is_defective"]:
        #     description = explain_defect(img_bytes)
        #     print("📝 Explanation:", description)
        # else:
        #     print("✅ Product appears to be undamaged.")
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
