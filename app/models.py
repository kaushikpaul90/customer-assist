# app/models.py
# import os
import torch
from transformers import AutoModelForCausalLM, Blip2ForConditionalGeneration, Blip2Processor, CLIPModel, CLIPProcessor, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import easyocr
import numpy as np
from PIL import Image
import io
import cv2
import re
import tempfile
from langdetect import detect
# import timm
# from dotenv import load_dotenv

from app.config import (
    ASR_MODEL, DAMAGE_DETECTOR, QA_MODEL, SUMMARIZER, EXPLAINER,
    PARAPHRASE_MODEL, TEXT_DAMAGE_CLASSIFIER, TRANSLATOR, IMG_CLASSIFIER,
    IMAGE_CAPTION, OCR_LANGS, DEVICE
)

# Load environment variables from .env file
# load_dotenv()

# ----------------------------
# Initialize NLP Pipelines
# ----------------------------
# QA
qa_pipeline = pipeline(
    "question-answering",
    model=QA_MODEL,
    tokenizer=QA_MODEL,
    device=0 if DEVICE == "cuda" else -1
)

# Summarization
summarizer = pipeline(
    "summarization",
    model=SUMMARIZER,
    tokenizer=SUMMARIZER,
    device=0 if DEVICE == "cuda" else -1
)

# # Explanation (text2text)
# explainer = pipeline(
#     "text2text-generation",
#     model=EXPLAINER,
#     tokenizer=EXPLAINER,
#     device=0 if DEVICE == "cuda" else -1
# )

# Explanation (text2text)
explainer = pipeline(
    "text-generation",
    model=EXPLAINER,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)
# Explanation (text2text)
# explainer = pipeline(
#     "text-generation",
#     model=EXPLAINER,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
#     token=HF_KEY
# )

# # Paraphrase / Simplify (T5)
# # we'll use a T5 text2text pipeline and control via prompt
# # paraphraser = pipeline(
# #     "text2text-generation",
# #     model=PARAPHRASE_MODEL,
# #     tokenizer=PARAPHRASE_MODEL,
# #     device=0 if DEVICE == "cuda" else -1
# # )
# paraphraser = pipeline(
#     "text-generation",
#     model=PARAPHRASE_MODEL,
#     tokenizer=PARAPHRASE_MODEL,
#     torch_dtype="auto",
#     device_map="auto"
# )

# ----------------------------
# Initialize CV Pipelines
# ----------------------------
# Image classification (cached)
img_classifier = pipeline(
    "image-classification",
    model=IMG_CLASSIFIER,
    device=0 if DEVICE == "cuda" else -1
)

# Image captioning (BLIP)
image_captioner = pipeline(
    "image-to-text",
    model=IMAGE_CAPTION,
    device=0 if DEVICE == "cuda" else -1
)

damage_detector = pipeline(
    "image-classification",
    model=DAMAGE_DETECTOR,
    device=0 if DEVICE == "cuda" else -1
)
# damage_detector = timm.create_model("hf_hub:Marqo/nsfw-image-detection-384", pretrained=True)

damage_text_classifier = pipeline(
    "text-classification",
    model=TEXT_DAMAGE_CLASSIFIER,
    device=0 if DEVICE == "cuda" else -1
)

# ----------------------------
# Initialize ASR Pipelines
# ----------------------------
asr_pipeline = pipeline("automatic-speech-recognition", model=ASR_MODEL, return_timestamps=True, device=-1)


use_gpu = torch.backends.mps.is_available()
# OCR reader (easyocr)
reader = easyocr.Reader(OCR_LANGS, gpu=use_gpu)

# ----------------------------
# Helpers
# ----------------------------
def split_text(text, max_words=300):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

# Basic image preprocessing for OCR
def preprocess_image_for_ocr(image_bytes: bytes):
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)

    # Convert to RGB if not already (handles RGBA, CMYK, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # image = np.array(image.convert("L"))
    image_np = np.array(image)

    return image_np

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur to reduce noise
    # blurred = cv2.GaussianBlur(image, (3, 3), 0)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive thresholding for binarization
    processed = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 2)
    return processed

# Simple text cleanup
def clean_text(t: str) -> str:
    t = t.replace("\n", " ").strip()
    t = re.sub(r'\s+', ' ', t)
    return t

def contains_damage_keywords(text: str) -> bool:
    """
    Common helper to check if text contains any damage-related keywords.
    """
    damage_keywords = ["broken", "damaged", "defect", "cracked", "torn", "scratched"]
    return any(k in text.lower() for k in damage_keywords)

# ----------------------------
# NLP Tasks
# ----------------------------
def answer_question(context: str, question: str):
    answers = []
    for chunk in split_text(context, 300):
        res = qa_pipeline(question=question, context=chunk)
        # pipeline returns dict with 'answer' key
        answers.append(res.get("answer", ""))
    if not answers:
        return {"question": question, "answer": ""}
    final_answer = max(set(answers), key=answers.count)
    return {"question": question, "answer": final_answer}

def summarize_text(text: str):
    result = summarizer(text, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return result[0]["summary_text"]

def explain_topic(topic: str, style: str = "detailed"):
    prompt = (
        f"Explain the topic '{topic}' in five to six sentences. "
        f"Include what it is, how it works, and why it matters. "
        f"Use multiple complete sentences and examples if possible."
    )
    out = explainer(
        prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.5
    )
    explanation = out[0].get("generated_text") or out[0].get("text") or ""
    explanation = clean_text(explanation.replace(prompt, ""))
    return explanation

    # tokenizer = AutoTokenizer.from_pretrained(EXPLAINER, use_auth_token=HF_KEY)
    # model = AutoModelForCausalLM.from_pretrained(
    #     EXPLAINER,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    #     use_auth_token=HF_KEY
    # )

    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # outputs = model.generate(
    #     **inputs,
    #     max_new_tokens=30,
    #     temperature=0.7,
    #     top_p=0.9,
    #     do_sample=True
    # )

    # topic = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # topic = topic.split("Topic:")[-1].strip()

    # return {"topic": topic}

# def paraphrase_text(text: str, mode: str = "paraphrase"):
#     """
#     Paraphrase or simplify input text.
#     mode: 'paraphrase' or 'simplify'
#     Uses prompt-style calls on a small T5 model.
#     """
#     text = text.strip()
#     if mode == "simplify":
#         prompt = f"simplify: {text}"
#     else:
#         prompt = f"paraphrase: {text}"

#     out = paraphraser(
#         prompt,
#         max_length=60,
#         num_beams=4,
#         do_sample=False,
#         repetition_penalty=1.2
#     )
#     res = out[0].get("generated_text") or out[0].get("text") or ""
#     return clean_text(res)

def translate_text(text: str, src_lang: str, target_lang: str):
    """
    Translate input text to target language.
    target_lang: language code, e.g., 'fr' for French, 'de' for German.
    """

    translator = pipeline(
        "translation",
        model=TRANSLATOR,
        tokenizer=TRANSLATOR,
        src_lang=src_lang,
        tgt_lang=target_lang,
        device=0 if DEVICE == "cuda" else -1
    )
    prompt = f"{text}"
    out = translator(prompt, max_length=512, do_sample=True)
    res = out[0].get("translation_text") or out[0].get("text") or ""
    return clean_text(res)

# ----------------------------
# Computer Vision Tasks
# ----------------------------
def ocr_image_bytes(image_bytes: bytes):
    processed = preprocess_image_for_ocr(image_bytes)
    texts = reader.readtext(processed, detail=0)
    combined = " ".join(texts).strip()
    combined = re.sub(r"[^A-Za-z0-9.,;:!?'\s]", "", combined)
    return {"text": combined if combined else "No English text detected."}

def classify_image_bytes(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    preds = img_classifier(image)
    top3 = preds[:3]
    formatted = [{"label": p["label"], "score": float(p["score"])} for p in top3]
    return {"predictions": formatted}

def image_caption_bytes(image_bytes: bytes):
    """
    Produce a short caption describing the image.
    Uses BLIP (Salesforce/blip-image-captioning-base) model via pipeline 'image-to-text'.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    out = image_captioner(image, max_new_tokens=512)
    caption = out[0].get("generated_text") or out[0].get("caption") or ""
    caption = clean_text(caption)
    return {"caption": caption}

def detect_damage(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    preds = damage_detector(image)

    # Example: classifier output [{'label': 'LABEL_123', 'score': 0.98}, ...]
    # You can refine this logic using fine-tuned models.
    label = preds[0]['label'].lower()
    score = preds[0]['score']

    # # simple heuristic for demo
    # damaged_keywords = ["broken", "damaged", "defect", "crack", "torn"]
    # is_damaged = any(k in label for k in damaged_keywords)

    is_damaged = contains_damage_keywords(label)

    return is_damaged

def infer_damage_from_caption(caption: str):
    if contains_damage_keywords(caption):
        return True
    # keywords = ["cracked", "broken", "damaged", "defective", "scratched", "torn"]
    # if any(k in caption.lower() for k in keywords):
    #     return True
    sentiment = damage_text_classifier(caption)[0]
    if sentiment["label"].upper() == "NEGATIVE" and sentiment["score"] > 0.8:
        return True
    return False

# -------------------------------------------------------------
# 1️⃣ Zero-shot Defect Detection using CLIP/SigLIP
# -------------------------------------------------------------
def detect_defect(image_bytes: bytes,
                  model_id: str = "openai/clip-vit-large-patch14",
                  diff_margin: float = 0.05,
                  top_threshold: float = 0.25
    ) -> dict:
    """
    Adaptive hybrid CLIP-based defect detector.
    Compares average group probabilities and top defective label confidence
    without relying on hard absolute thresholds.
    """

    # Load model & processor
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Define prompt groups
    good_labels = [
        "a brand new product with no defects",
        "a perfect product in good condition",
        "an undamaged and clean product",
        "a functional product that looks new",
        "a product without any visible defects"
    ]

    defective_labels = [
        "a broken product",
        "a damaged product",
        "a defective product",
        "a scratched product",
        "a torn product",
        "a chipped product",
        "a dented product",
        "a bent product",
        "a stained product",
        "a product with missing parts",
        "a tampered product",
        "a discolored or faded product",
        "a dirty or moldy product",
        "a crushed or warped product",
        "a product with visible loose threads",
        "a product with frayed seams",
        # "a product with loose stitching",
        "a product with holes or rips"
    ]

    all_labels = good_labels + defective_labels

    # Preprocess
    inputs = processor(text=all_labels, images=image, return_tensors="pt", padding=True)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    # Map scores
    result = dict(zip(all_labels, probs.tolist()))

    # # Get top prediction
    # top_label = max(result, key=result.get)
    # top_prob = result[top_label]

    # # Identify if defective based on top label
    # is_defective = not any(x in top_label for x in ["brand new", "no defects"]) and top_prob > threshold

    # return {
    #     "is_defective": is_defective,
    #     "predicted_label": top_label,
    #     "confidence": round(top_prob, 3)
    #     # "scores": {k: round(v, 3) for k, v in result.items()}
    # }

    # Group-wise averages
    avg_good = sum(result[l] for l in good_labels) / len(good_labels)
    avg_def = sum(result[l] for l in defective_labels) / len(defective_labels)

    # Top defective label
    top_def_label = max(defective_labels, key=lambda l: result[l])
    top_def_prob = result[top_def_label]

    # Adaptive decision logic
    diff = avg_def - avg_good

    # # 7️⃣ Find top label and probability
    # top_label = max(result, key=result.get)
    # top_prob = result[top_label]

    # Decision logic
    is_defective = False

    # Case 1: Strongly defective
    if top_def_prob > top_threshold:
        is_defective = True

    # Case 2: Weak but consistent defect signals
    elif avg_def > avg_good and diff > 0.02:  # smaller diff margin
        is_defective = True

    # Case 3: borderline but top defective probability still reasonably high
    elif avg_def > avg_good and top_def_prob > 0.15:
        is_defective = True

    return {
        "is_defective": is_defective,
        # "predicted_label": top_def_label,
        "confidence": float(top_def_prob),
        # "diff": float(diff),
        # "avg_def": float(avg_def),
        # "avg_good": float(avg_good)
    }

# -------------------------------------------------------------
# 2️⃣ Optional: Defect Explanation using BLIP-2
# -------------------------------------------------------------
def explain_defect(image_bytes: bytes,
                   model_id: str = "Salesforce/blip2-flan-t5-xl",
                   max_tokens: int = 60) -> str:
    """
    Generates a natural-language description of any visible defect.

    Args:
        image_path: Path to the uploaded image.
        model_id: BLIP-2 model identifier.
        max_tokens: Max length of the explanation.

    Returns:
        str: Textual description of the defect.
    """
    
    device = "mps" if torch.mps.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prompt = "Mention any visible defects or damages in this product image."

    inputs = processor(image, prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_tokens)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption

# --- Speech Recognition Task ---

def transcribe_audio(audio_bytes: bytes):
    """Convert spoken audio to text using Whisper model."""
    # Save to temporary WAV for pipeline compatibility
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = asr_pipeline(tmp.name)
    return {"transcription": result["text"]}


def transcribe_and_translate_audio(audio_bytes: bytes):
    """Convert spoken audio to text using Whisper model and translate to English text."""
    transcribed_text = transcribe_audio(audio_bytes)
    src_lang, target_lang = detect_source_target_language(transcribed_text["transcription"])
    translated_text = translate_text(text=transcribed_text["transcription"], src_lang=src_lang, target_lang=target_lang)
    return translated_text

def detect_source_target_language(transcribed_text: str):
    """Detect the source and target language of the spoken audio."""
    transcribed_language_code = detect(transcribed_text)
    if transcribed_language_code == 'hi':
        src_lang = 'hin_Deva'
        target_lang = 'eng_Latn'
    else:
        src_lang = 'eng_Latn'
        target_lang = 'hin_Deva'
    return src_lang, target_lang