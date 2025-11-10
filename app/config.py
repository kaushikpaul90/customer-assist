"""Configuration constants used across the demo app.

This module exposes a small list of model identifiers and path
constants. These are intentionally simple constants (strings and
Paths) so callers can import them without triggering heavy work.
"""

from pathlib import Path
import torch

# Directory paths referenced by other modules (not created here).
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

# Models (Hugging Face names) -------------------------------------------------
QA_MODEL = "deepset/roberta-base-squad2"
SUMMARIZER = "philschmid/bart-large-cnn-samsum"
EXPLAINER = "google/flan-t5-large"
TRANSLATOR = "facebook/nllb-200-distilled-600M"

# For paraphrasing / simplification we use a small T5 prompt approach
PARAPHRASE_MODEL = "google-t5/t5-small"

# Image models
IMG_CLASSIFIER = "google/vit-base-patch16-224"
IMAGE_CAPTION = "Salesforce/blip-image-captioning-base"
DAMAGE_DETECTOR = "microsoft/resnet-50"
TEXT_DAMAGE_CLASSIFIER = "distilbert-base-uncased-finetuned-sst-2-english"

# ASR (Whisper style model id)
ASR_MODEL = "openai/whisper-medium"

# OCR languages (easyocr)
OCR_LANGS = ["en"]

# Compute device (string) - used by pipelines to set device flags
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
