"""Configuration constants used across the demo app.

This module exposes a small list of model identifiers and path
constants. These are intentionally simple constants (strings and
Paths) so callers can import them without triggering heavy work.
"""

import torch
import logging

# --- LOGGING CONFIGURATION ---
LOG_FORMAT = '%(levelname)s %(asctime)s %(name)s %(message)s'
LOG_LEVEL = logging.INFO

# --- LLMOPS: Simple Model Registry ---
# Define a dictionary for each model family to manage versions.
# The 'active' key determines the currently used model version.

MODEL_REGISTRY = {
    "qa_model": {
        "v1": "deepset/roberta-base-squad2",
        "active": "v1"
    },
    "summarizer": {
        "v1": "philschmid/bart-large-cnn-samsum",
        "active": "v1"
    },
    "translator": {
        "v1": "facebook/nllb-200-distilled-600M",
        "active": "v1"
    },
    "img_classifier": {
        "v1": "google/vit-base-patch16-224",
        "active": "v1"
    },
    "image_caption": {
        "v1": "Salesforce/blip-image-captioning-base",
        "active": "v1"
    },
    "defect_detector": {
        "v1": "openai/clip-vit-large-patch14",
        "active": "v1"
    },
    "asr_model": {
        "v1": "openai/whisper-medium",
        "active": "v1"
    }
}

# Helper function to get the currently active model ID
def get_model_id(model_key: str) -> str:
    """Returns the Hugging Face ID for the active version of a model."""
    registry = MODEL_REGISTRY.get(model_key)
    if not registry:
        raise ValueError(f"Model key '{model_key}' not found in registry.")
    version = registry['active']
    return registry[version]

# --- LLMOPS: Prompt Registry (New) ---
PROMPT_REGISTRY = {
    "summarizer_prompt": {
        "v1": (
            "Summarize the following customer service conversation. "
            "Focus on the customer's issue, agent's response, resolution offered, and any constraints mentioned:\n\n"
        ),
        "v2": (
            "Provide a concise, neutral summary of the key events, customer intent, and final resolution. "
            "Use bullet points for clarity:\n\n"
        ),
        "active": "v1" # This is the version currently in use
    },
    "translator_prompt": {
        "v1": "{text}", # Simple wrapper prompt
        "active": "v1"
    }
}

# Helper function to get the currently active prompt template (New)
def get_prompt_template(prompt_key: str) -> tuple[str, str]:
    """
    Returns the active prompt template string and its version.
    Returns: (template_string, version_string)
    """
    registry = PROMPT_REGISTRY.get(prompt_key)
    if not registry:
        raise ValueError(f"Prompt key '{prompt_key}' not found in registry.")
    version = registry['active']
    return registry[version], version

# Models (Hugging Face names) -------------------------------------------------
QA_MODEL = get_model_id("qa_model")
SUMMARIZER = get_model_id("summarizer")
TRANSLATOR = get_model_id("translator")

# Image models
IMG_CLASSIFIER = get_model_id("img_classifier")
IMAGE_CAPTION = get_model_id("image_caption")
DEFECT_DETECTOR = get_model_id("defect_detector")

# ASR (Whisper style model id)
ASR_MODEL = get_model_id("asr_model")

# OCR languages (easyocr)
OCR_LANGS = ["en"]

# Compute device (string) - macOS uses MPS (Metal Performance Shaders) for GPU,
# falls back to CPU otherwise. CUDA is not available on macOS.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ----------------------------
# Pipeline Task Names
# ----------------------------
TASK_QA = "question-answering"
TASK_SUMMARIZATION = "summarization"
TASK_TEXT_GENERATION = "text-generation"
TASK_IMAGE_CLASSIFICATION = "image-classification"
TASK_IMAGE_CAPTION = "image-to-text"
TASK_TRANSLATION = "translation"
TASK_ASR = "automatic-speech-recognition"

# ----------------------------
# Pipeline Device Configuration
# ----------------------------
# Note: transformers pipeline() uses -1 for CPU and 0 for GPU
# On macOS with MPS, we use CPU device (-1) as transformers doesn't support MPS directly
DEVICE_ID = -1  # CPU device for transformers pipelines
DEVICE_MAP = "cpu"  # Use CPU device mapping for transformers models

# ----------------------------
# Pipeline Keyword Arguments
# ----------------------------
QA_PIPELINE_KWARGS = {
    "device": DEVICE_ID
}

SUMMARIZER_PIPELINE_KWARGS = {
    "device": DEVICE_ID
}

IMG_CLASSIFIER_PIPELINE_KWARGS = {
    "device": DEVICE_ID
}

IMAGE_CAPTIONER_PIPELINE_KWARGS = {
    "device": DEVICE_ID
}

ASR_PIPELINE_KWARGS = {
    "return_timestamps": True,
    "device": DEVICE_ID  # Use CPU device on macOS
}