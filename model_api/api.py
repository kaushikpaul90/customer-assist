# File: model_api/api.py

import torch
import re
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path

# Import the model manager
import model_manager

# --- Pydantic Models for Request/Response ---

class ChatRequest(BaseModel):
    human_input: str

class ChatResponse(BaseModel):
    assistant_output: str

# --- FastAPI Lifespan Event (Replaces on_event) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup and shutdown events.
    On startup, it loads the model.
    """
    print("API starting up...")
    
    # --- Define your model paths ---
    BASE_MODEL = "distilgpt2"
    
    # This script is in 'model_api/', so it goes up one level ('..') 
    # and looks in the 'models/' directory.
    # ADAPTER_DIR = "../models/distilgpt2_finetuned_adapter"
    BASE_DIR = Path(__file__).resolve().parent.parent  # goes up one level
    ADAPTER_DIR = BASE_DIR / "models" / "distilgpt2_finetuned_adapter"

    model_manager.load_model(
        base_model_id=BASE_MODEL,
        adapter_dir=ADAPTER_DIR
    )
    yield
    print("API shutting down...")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Customer Support API",
    description="An API to serve the distilgpt2 fine-tuned model.",
    lifespan=lifespan
)

# --- Generation Logic ---

def get_assistant_reply(human_query: str) -> str:
    """
    Generates a response using the loaded model.
    """
    if not model_manager.model or not model_manager.tokenizer:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    prompt = f"Human: {human_query}\nAssistant:"
    
    enc = model_manager.tokenizer(
        prompt, 
        return_tensors="pt", 
        return_attention_mask=True
    ).to(model_manager.device)

    # Generation parameters
    gen_kwargs = dict(
        max_new_tokens=80,
        do_sample=True,
        temperature=0.22,
        top_k=40,
        eos_token_id=model_manager.tokenizer.eos_token_id,
        pad_token_id=model_manager.tokenizer.pad_token_id,
        use_cache=True,
    )

    with torch.no_grad():
        out = model_manager.model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],  # Pass attention_mask
            **gen_kwargs
        )

    # Decode only the newly generated tokens
    gen_tokens = out[0][enc["input_ids"].shape[-1]:]
    raw = model_manager.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # (Your sanitization logic from the notebook can be added here if needed)
    
    # Cut at potential echoes
    reply = re.split(r"\n\s*\n|Human:|Assistant:", raw)[0].strip()

    if not reply.endswith((".","!","?")):
        reply = reply + "."

    return reply

# --- API Endpoint ---

@app.post("/v1/qa-finetuned-model", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    The main conversation endpoint.
    Receives a 'human_input' and returns an 'assistant_output'.
    """
    try:
        response_text = get_assistant_reply(request.human_input)
        return ChatResponse(assistant_output=response_text)
    except RuntimeError as e:
        # Catch potential 'nan' errors from unstable generation
        if 'nan' in str(e) or 'inf' in str(e):
            print(f"Generation Error (nan/inf): {e}")
            raise HTTPException(status_code=500, detail="Model generation failed (nan/inf).")
        else:
            print(f"Error during generation: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Run the API (for testing) ---
if __name__ == "__main__":
    # Note: The port is 8000 here, but your log shows 8010. 
    # Uvicorn will default to 8000 if not specified.
    uvicorn.run(app, host="127.0.0.1", port=8000)