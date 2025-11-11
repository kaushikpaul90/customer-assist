"""app.models

High-level ML helper wrappers used by the Customer Assist API.

This module centralizes lightweight helper functions and a small set of
pre-initialized Hugging Face `pipeline` objects used by the FastAPI
endpoints. Note: importing this module will initialize several
transformers pipelines which may download models and allocate device memory.
Keep import-time work minimal in production (consider lazy-loading).
"""

import torch
from transformers import CLIPModel, CLIPProcessor, pipeline
import easyocr
import numpy as np
from PIL import Image
import io
import re
import tempfile
from langdetect import detect
from app.config import (
    QA_MODEL,
    SUMMARIZER,
    TRANSLATOR,
    IMG_CLASSIFIER,
    IMAGE_CAPTION,
    ASR_MODEL,
    DEFECT_DETECTOR,
    OCR_LANGS,
    DEVICE,
    TASK_QA,
    TASK_SUMMARIZATION,
    TASK_TEXT_GENERATION,
    TASK_IMAGE_CLASSIFICATION,
    TASK_IMAGE_CAPTION,
    TASK_TRANSLATION,
    TASK_ASR,
    QA_PIPELINE_KWARGS,
    SUMMARIZER_PIPELINE_KWARGS,
    IMG_CLASSIFIER_PIPELINE_KWARGS,
    IMAGE_CAPTIONER_PIPELINE_KWARGS,
    ASR_PIPELINE_KWARGS,
    get_prompt_template
)

# ==============================================================================
# 1. NLP PIPELINES & FUNCTIONS
# ==============================================================================

# ----------------------------
# NLP Pipeline Initialization
# ----------------------------
# The pipelines below are created at import time for convenience. They
# are convenient for a demo, but they are heavyweight: model downloads,
# tokenizer initialization and device memory allocation can occur during
# import. For production or tests consider lazy factories (functions that
# create pipelines on first call) to speed up cold-starts and reduce
# resource usage in environments that only run limited endpoints.

# QA Pipeline
qa_pipeline = pipeline(
    TASK_QA,
    model=QA_MODEL,
    tokenizer=QA_MODEL,
    **QA_PIPELINE_KWARGS
)

# Summarization Pipeline
summarizer = pipeline(
    TASK_SUMMARIZATION,
    model=SUMMARIZER,
    tokenizer=SUMMARIZER,
    **SUMMARIZER_PIPELINE_KWARGS
)

# ----------------------------
# NLP Helper Functions
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


def clean_text(t: str) -> str:
    """Normalize and trim whitespace in model-generated text.

    Removes newlines and collapses repeated whitespace so downstream
    code can rely on a single-line string for display or further
    processing.
    """

    t = t.replace("\n", " ").strip()
    t = re.sub(r'\s+', ' ', t)
    return t


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
        tuple: (generated_summary_str, full_prompt_str, prompt_version_str)
    """

    # LLMOPS: Fetch the active prompt template and version
    template, version = get_prompt_template("summarizer_prompt")

    # Structured prompt to guide the model
    prompt = template + text.strip()

    # Generate summary with deterministic decoding
    result = summarizer(
        prompt,
        max_new_tokens=160,  # Allow more room for nuance
        do_sample=False,     # Disable sampling for consistency
        num_beams=4,         # Beam search for better quality
        early_stopping=True
    )

    summary = result[0]["summary_text"]
    return summary, prompt, version # LLMOPS: Return prompt and version


def translate_text(text: str, src_lang: str, target_lang: str):
    """Translate input text to target language.

    Args:
        text: The input text to translate.
        src_lang: Source language code (e.g., 'eng_Latn' for English).
        target_lang: Target language code (e.g., 'hin_Deva' for Hindi).

    Returns:
        tuple: (translated_text_str, full_prompt_str, prompt_version_str)
    """

    # LLMOPS: Fetch the active prompt template and version
    template, version = get_prompt_template("translator_prompt")
    
    translator = pipeline(
        TASK_TRANSLATION,
        model=TRANSLATOR,
        tokenizer=TRANSLATOR,
        src_lang=src_lang,
        tgt_lang=target_lang,
        device=0 if DEVICE == "cuda" else -1
    )
    # The prompt for translation is just the input text itself using v1 template
    prompt = template.format(text=text)
    out = translator(prompt, max_length=512, do_sample=True)
    res = out[0].get("translation_text") or out[0].get("text") or ""
    return clean_text(res), prompt, version # LLMOPS: Return prompt and version

# ==============================================================================
# 2. COMPUTER VISION PIPELINES & FUNCTIONS
# ==============================================================================

# ----------------------------
# Computer Vision Pipeline Initialization
# ----------------------------

# Image Classification Pipeline
img_classifier = pipeline(
    TASK_IMAGE_CLASSIFICATION,
    model=IMG_CLASSIFIER,
    **IMG_CLASSIFIER_PIPELINE_KWARGS
)

# Image Captioning Pipeline (BLIP)
image_captioner = pipeline(
    TASK_IMAGE_CAPTION,
    model=IMAGE_CAPTION,
    **IMAGE_CAPTIONER_PIPELINE_KWARGS
)

# OCR Reader (easyocr)
use_gpu = torch.backends.mps.is_available()
reader = easyocr.Reader(OCR_LANGS, gpu=use_gpu)

# ----------------------------
# Computer Vision Helper Functions
# ----------------------------

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

    Args:
        image_bytes: Raw image bytes.

    Returns:
        dict: {'predictions': [{'label': str, 'score': float}, ...]}
    """

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    preds = img_classifier(image)
    top3 = preds[:3]
    formatted = [{"label": p["label"], "score": float(p["score"])} for p in top3]
    return {"predictions": formatted}


def detect_defect(image_bytes: bytes,
                  model_id: str = DEFECT_DETECTOR,
                  diff_margin: float = 0.05,
                  top_threshold: float = 0.25
    ) -> dict:
    """Adaptive hybrid CLIP-based defect detector.

    Compares average group probabilities and top defective label confidence
    without relying on hard absolute thresholds. Uses zero-shot classification
    to detect product defects without fine-tuning.

    Args:
        image_bytes: Raw image bytes to analyze.
        model_id: CLIP model identifier.
        diff_margin: Margin for difference between defective and good scores.
        top_threshold: Confidence threshold for top defective label.

    Returns:
        dict: {'is_defective': bool, 'confidence': float}
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
        "predicted_label": top_def_label,
        "eligible_for_return": is_defective
    }

# ==============================================================================
# 3. AUDIO & SPEECH RECOGNITION PIPELINES & FUNCTIONS
# ==============================================================================

# ----------------------------
# Audio Pipeline Initialization
# ----------------------------

# Automatic Speech Recognition Pipeline (Whisper)
asr_pipeline = pipeline(TASK_ASR, model=ASR_MODEL, **ASR_PIPELINE_KWARGS)

# ----------------------------
# Audio Helper Functions
# ----------------------------

def transcribe_audio(audio_bytes: bytes):
    """Convert spoken audio to text using Whisper model.

    Args:
        audio_bytes: Raw audio file bytes (WAV, MP3, etc.).

    Returns:
        dict: {'transcription': str} containing the recognized speech text.
    """
    # Save to temporary WAV for pipeline compatibility
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = asr_pipeline(tmp.name)
    return {"transcription": result["text"]}


def detect_source_target_language(transcribed_text: str):
    """Detect the source and target language of the spoken audio.

    Uses langdetect to identify the language of transcribed text,
    then maps it to translation language codes.

    Args:
        transcribed_text: The transcribed text to analyze.

    Returns:
        tuple: (src_lang, target_lang) as NLLB-200 language codes.
    """
    transcribed_language_code = detect(transcribed_text)
    if transcribed_language_code == 'hi':
        src_lang = 'hin_Deva'
        target_lang = 'eng_Latn'
    else:
        src_lang = 'eng_Latn'
        target_lang = 'hin_Deva'
    return src_lang, target_lang


def transcribe_and_translate_audio(audio_bytes: bytes):
    """Convert spoken audio to text and translate to target language.

    First transcribes audio to text using Whisper, detects the source
    language, then translates to the target language.

    Args:
        audio_bytes: Raw audio file bytes.

    Returns:
        str: Translated text.
    """
    transcribed_text = transcribe_audio(audio_bytes)
    src_lang, target_lang = detect_source_target_language(transcribed_text["transcription"])
    translated_text, _, _ = translate_text(text=transcribed_text["transcription"], src_lang=src_lang, target_lang=target_lang)
    return translated_text
