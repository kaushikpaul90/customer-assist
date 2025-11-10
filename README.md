# CustomerAssist - Customer Service AI Assistant (Minimal Distribution)

This package contains a minimal API demonstrating an customer service focused AI assistant.

## What's included
- FastAPI server with endpoints: /qa, /summarize, /explain, /upload-image
- Lightweight model wrappers using Hugging Face pipelines
- Fine-tune and evaluation scripts for QA (SQuAD subset)

## Getting started (local)
1. Create venv and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or venv\\Scripts\\activate on Windows
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
3. Test with curl or Postman (examples in the earlier project doc).

## Colab
Use the provided `colab/customerAssist_finetune.ipynb` to run fine-tuning on Colab GPU.
