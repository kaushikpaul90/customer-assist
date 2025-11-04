from pathlib import Path
import torch

# ROOT = Path(__file__).parent.parent
# MODEL_DIR = ROOT / "models"
# DATA_DIR = ROOT / "data"
# QA_MODEL = "distilbert-base-cased-distilled-squad"
# SUMMARIZER = "google/flan-t5-small"
# EXPLAINER = "google/flan-t5-small"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory paths
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

# Models (Hugging Face names)
QA_MODEL = "deepset/roberta-base-squad2"
SUMMARIZER = "Falconsai/text_summarization" #"facebook/bart-large-cnn"
EXPLAINER = "google/flan-t5-large"  #"meta-llama/Meta-Llama-3-8B"
TRANSLATOR = "facebook/nllb-200-distilled-600M"

# For paraphrasing / simplification we use T5-small prompt approach
PARAPHRASE_MODEL = "google-t5/t5-small"

# Image models
IMG_CLASSIFIER = "google/vit-base-patch16-224"
IMAGE_CAPTION = "Salesforce/blip-image-captioning-base"    #"datalab-to/chandra"
DAMAGE_DETECTOR = "microsoft/resnet-50" #"Marqo/nsfw-image-detection-384"
TEXT_DAMAGE_CLASSIFIER = "distilbert-base-uncased-finetuned-sst-2-english"    #"mrm8488/codebert2codebert-finetuned-code-defect-detection"
ASR_MODEL = "openai/whisper-medium"

# OCR
OCR_LANGS = ["en"]

# Compute device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
