"""app.models

High-level ML helper wrappers used by the Customer Assist API.

This module centralizes lightweight helper functions and a small set of
pre-initialized Hugging Face `pipeline` objects used by the FastAPI
endpoints. Note: importing this module will initialize several
transformers pipelines which may download models and allocate device memory.
Keep import-time work minimal in production (consider lazy-loading).
"""

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
# The pipelines below are created at import time for convenience. They
# are convenient for a demo, but they are heavyweight: model downloads,
# tokenizer initialization and device memory allocation can occur during
# import. For production or tests consider lazy factories (functions that
# create pipelines on first call) to speed up cold-starts and reduce
# resource usage in environments that only run limited endpoints.
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

# Explanation (text2text)
explainer = pipeline(
    "text-generation",
    model=EXPLAINER,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

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
    """Yield successive chunks of text limited by `max_words`.

    This is a simple splitter used to avoid sending very long
    documents to QA/summarization pipelines in a single call.

    Args:
        text: input string to split.
        max_words: approximate maximum words per yielded chunk.

    Yields:
        str: chunk of the original text (<= max_words words).
    """

    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

# Basic image preprocessing for OCR
def preprocess_image_for_ocr(image_bytes: bytes):
    """Basic image preprocessing for OCR pipelines.

    Converts raw image bytes to a NumPy array in RGB format which
    `easyocr.Reader` can consume. This function intentionally keeps the
    transformations small (no binarization or heavy denoising) to be
    robust across many input images. If you need more aggressive
    preprocessing (e.g. thresholding) move that logic into a separate
    preprocessing pipeline and test on sample inputs.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        np.ndarray: RGB image as a NumPy array.
    """

    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)

    # Convert to RGB if not already (handles RGBA, CMYK, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_np = np.array(image)
    return image_np

# Simple text cleanup
def clean_text(t: str) -> str:
    """Normalize and trim whitespace in model-generated text.

    Removes newlines and collapses repeated whitespace so downstream
    code can rely on a single-line string for display or further
    processing.
    """

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
    """Answer a question using the QA pipeline over possibly-long context.

    The function will split long context into smaller chunks (using
    `split_text`) and query the QA pipeline for each chunk. The final
    answer is chosen by simple majority voting across chunk-level
    responses. This strategy is lightweight and suitable for short to
    medium documents. For better accuracy consider a more advanced
    retrieval + reading pipeline.

    Args:
        context: Long textual context to search for the answer.
        question: The question string to answer.

    Returns:
        dict: A dictionary with keys `question` and `answer`.
    """

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
    """Summarize a customer service conversation or long text.

    The implementation builds a short prompt to focus the summarizer on
    the customer's issue and the resolution. The summarizer is called
    with deterministic decoding parameters for consistent outputs.

    Args:
        text: The input text to summarize.

    Returns:
        str: The generated summary.
    """

    # Structured prompt to guide the model
    prompt = (
        "Summarize the following customer service conversation. "
        "Focus on the customer's issue, agent's response, resolution offered, and any constraints mentioned:\n\n"
        + text.strip()
    )

    # Generate summary with deterministic decoding
    result = summarizer(
        prompt,
        max_new_tokens=160,  # Allow more room for nuance
        do_sample=False,     # Disable sampling for consistency
        num_beams=4,         # Beam search for better quality
        early_stopping=True
    )

    summary = result[0]["summary_text"]
    return summary

def explain_topic(topic: str, style: str = "detailed"):
    """Return a concise explanation of `topic`.

    Args:
        topic: The subject to explain.
        style: A non-strict hint for the explanation style (e.g.
            'detailed' or 'step-by-step'). Currently the value is passed
            through but not used to change the prompt substantially.

    Returns:
        str: Cleaned explanation text.
    """

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
    """Run OCR on image bytes and return combined plain text.

    Uses `easyocr.Reader` which is initialized at module import with
    languages from `config.OCR_LANGS`.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        dict: {'text': recognized_text} or a message when nothing found.
    """

    processed = preprocess_image_for_ocr(image_bytes)
    texts = reader.readtext(processed, detail=0)
    combined = " ".join(texts).strip()
    combined = re.sub(r"[^A-Za-z0-9.,;:!?'\s]", "", combined)
    return {"text": combined if combined else "No English text detected."}

def classify_image_bytes(image_bytes: bytes):
    """Return top-3 image classification predictions.

    The function uses a pre-initialized `img_classifier` pipeline and
    returns the top 3 labels with scores as floats for JSON
    serialization.
    """

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
