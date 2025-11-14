import os
import torch
import re
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path

# --- 1. Settings ---
MODEL_ID = "distilgpt2"
DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "models" / "distilgpt2_finetuned_adapter"

EPOCHS = 3
BATCH_SIZE = 1 
MAX_LENGTH = 256

"""
This script demonstrates how to fine-tune a Small Language Model (SLM) 
(distilgpt2) on a customer support dataset using PEFT (LoRA) from
the Hugging Face library.

The script covers the entire pipeline:
1. Configuration and Setup
2. Loading the Tokenizer
3. Loading and formatting the Dataset
4. Loading the Base Model
5. Applying the LoRA Adapter
6. Tokenizing the dataset for the trainer
7. Setting up and running the Trainer
8. Saving the trained adapter
9. Loading and testing the fine-tuned model for inference
"""

import os
import torch
import re
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path

# ==============================================================================
# === 1. CONFIGURATION
# ==============================================================================

# --- Model & Dataset Settings ---
MODEL_ID = "distilgpt2"
DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# --- Training & Hardware Settings ---
EPOCHS = 3
BATCH_SIZE = 1 
MAX_LENGTH = 256
USE_LORA = True  # Low-Rank Adaptation (LoRA) set to False to train the full model (much slower)
# LoRA - To adapt large, pre-trained machine learning models (like Large Language Models or diffusion models) to specific tasks or domains efficiently, with significantly reduced computational resources and memory compared to full fine-tuning.

# --- Filesystem Settings ---
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "models" / "distilgpt2_finetuned_adapter"

print(f"--- Settings ---")
print(f"MODEL_ID: {MODEL_ID}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"---")

# ==============================================================================
# === 2. DEVICE SETUP
# ==============================================================================

# Disable tokenizer parallelism to avoid issues with fork
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Select device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    device = 'cuda'
elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Set model dtype and pin_memory based on device
model_dtype = torch.float32 
use_pin_memory = True if device == 'cuda' else False

print(f'Using device: {device} (model dtype={model_dtype})')

# ==============================================================================
# === 3. TOKENIZER LOADING
# ==============================================================================
# We load the tokenizer first to handle special tokens
# (pad, eos) before processing the dataset.

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) 

# distilgpt2 (and gpt2) do not have a default pad token.
# We add one to enable padding, which is necessary for batched training.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

# Store the End-Of-Sequence token for formatting the dataset
eos_token = tokenizer.eos_token
print(f"Tokenizer loaded. EOS token: {eos_token}")


# ==============================================================================
# === 4. DATASET PREPARATION
# ==============================================================================

def safe_load_customer_dataset(dataset_id):
    """
    Tries to load the specified dataset. If it fails (e.g., no internet),
    it falls back to a small, hard-coded synthetic dataset for testing.
    """
    try:
        ds = load_dataset(dataset_id)
        # Ensure we get the 'train' split, or the only split available
        return ds.get('train', ds)
    except Exception as e:
        print(f'Could not load dataset {dataset_id}: {e}')
        print('Falling back to a tiny synthetic customer support dataset.')
        samples = [
            {'customer': "My order hasn't arrived, it's been 10 days.", 'agent': "I'm sorry. Can you share your order id?"},
            {'customer': 'I was charged twice for the same order.', 'agent': "I can help. Please share the transaction id."},
        ]
        return Dataset.from_list(samples)

def build_prompt_for_training(row):
    """
    Formats a single dataset row into a standardized prompt format.
    We append the EOS token to the agent's response to teach the
    model when to stop generating text.
    """
    # Check for different possible column names
    if 'customer' in row and 'agent' in row:
        text = f"Human: {row['customer']}\nAssistant: {row['agent']}{eos_token}\n"
    elif 'input' in row and 'output' in row:
        text = f"Human: {row['input']}\nAssistant: {row['output']}{eos_token}\n"
    elif 'instruction' in row and 'response' in row:
         text = f"Human: {row['instruction']}\nAssistant: {row['response']}{eos_token}\n"
    else:
        # Fallback for unknown format
        text = str(row)
    
    return {'text': text}

# --- Load, Format, and Split ---
print("Loading and preparing dataset...")
raw_ds = safe_load_customer_dataset(DATASET_ID)
ds = raw_ds.map(build_prompt_for_training)

# Split the dataset and select a subset for faster local runs
if len(ds) > 2000:
    # Use a small validation set and cap training samples
    ds_split = ds.train_test_split(test_size=0.05, shuffle=True, seed=42)
    train_ds = ds_split['train'].select(range(min(4096, len(ds_split['train']))))
    eval_ds = ds_split['test'].select(range(min(128, len(ds_split['test']))))
else:
    # Use a simple 90/10 split for very small datasets
    ds_split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = ds_split['train']
    eval_ds = ds_split['test']

print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

# ==============================================================================
# === 5. MODEL LOADING
# ==============================================================================
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=False,
    torch_dtype=model_dtype,
    low_cpu_mem_usage=True,
    attn_implementation="eager" # Use "eager" for broad compatibility
)

# CRITICAL: Resize token embeddings
# We must do this *after* loading the model and *before* applying PEFT
# to account for the new '<|pad|>' token we added to the tokenizer.
model.resize_token_embeddings(len(tokenizer))
print("Base model loaded and token embeddings resized.")

# ==============================================================================
# === 6. PEFT (LORA) SETUP
# ==============================================================================
if USE_LORA:
    print("Applying LoRA adapter (PEFT)...")
    lora_config = LoraConfig(
        r=8,                # Rank of the update matrices
        lora_alpha=16,      # Scaling factor
        # Target modules for distilgpt2's attention blocks
        target_modules=["c_attn", "c_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("LoRA adapter applied.")
    model.print_trainable_parameters() # Optional: Show % of params being trained
else:
    print("Skipping LoRA. Training full model (slower).")

# ==============================================================================
# === 7. TOKENIZATION FOR TRAINING
# ==============================================================================
def tokenize_for_lm(examples):
    """Tokenizes a batch of texts for Causal LM training."""
    # Extract texts from the 'text' column
    texts = [str(t) for t in examples.get('text', [])]
    
    # Tokenize
    outputs = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length" # Pad to max_length for consistent tensor sizes
    )
    
    # For Causal LM, the 'labels' are the same as 'input_ids'.
    # The model learns to predict the next token.
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

print("Tokenizing datasets...")
train_tok = train_ds.map(
    tokenize_for_lm, 
    batched=True, 
    remove_columns=train_ds.column_names # Remove old columns to save memory
)
eval_tok = eval_ds.map(
    tokenize_for_lm, 
    batched=True, 
    remove_columns=eval_ds.column_names
)

# Data Collator for Causal LM
# This will handle batching and (if needed) mask language modeling (mlm=False)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==============================================================================
# === 8. TRAINER SETUP
# ==============================================================================
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy='epoch',      # Evaluate at the end of each epoch
    save_strategy='epoch',      # Save at the end of each epoch
    logging_steps=10,
    save_total_limit=2,         # Keep only the last 2 checkpoints
    fp16=False,                 # Set to True if using CUDA and T4/V100+
    remove_unused_columns=False,
    dataloader_pin_memory=use_pin_memory,
    use_mps_device=(device == 'mps'), # Explicitly enable MPS device
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=data_collator,
)

# ==============================================================================
# === 9. TRAINING & SAVING
# ==============================================================================
print("--- Starting Training ---")
trainer.train()
print("--- Training Complete ---")

print(f"Saving fine-tuned adapter and tokenizer to {OUTPUT_DIR}...")
# This saves the LoRA adapter weights, not the full model
trainer.save_model(OUTPUT_DIR)
# Save the tokenizer so we load the correct special tokens for inference
tokenizer.save_pretrained(OUTPUT_DIR)
print("Artifacts saved successfully.")


# ==============================================================================
# === 10. INFERENCE (TESTING)
# ==============================================================================
print("\n--- Testing the Fine-Tuned Model ---")

# --- Clear memory before loading new model ---
del model
del trainer
if device == 'cuda':
    torch.cuda.empty_cache()
elif device == 'mps':
    torch.mps.empty_cache()

# --- Reloading process for PEFT adapters ---
# 1. Load the base model (in float32 for CPU/MPS, or original precision)
print("1. Loading base model for inference...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=False,
    torch_dtype=model_dtype,
    attn_implementation="eager"
)

# 2. Load the tokenizer *from the output directory*
print("2. Loading tokenizer from saved directory...")
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

# 3. Resize token embeddings (must match the training setup)
base_model.resize_token_embeddings(len(tokenizer))

# 4. Load the PEFT adapter and merge
print("3. Applying saved LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    OUTPUT_DIR,
    torch_dtype=model_dtype,
)

# 5. Move to device and set to eval mode
model = model.to(device)
model.eval()

print("Fine-tuned model reloaded for testing.")

# --- Test Generation ---
prompt = "Human: I haven't received my refund after 10 days. What should I do?\nAssistant:"
enc = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device)

# Generation parameters
gen_kwargs = dict(
    max_new_tokens=80,      # Max tokens to generate
    do_sample=True,         # Enable sampling
    temperature=0.5,        # Controls "creativity" (lower is more deterministic)
    top_k=40,               # Considers only the top K most likely tokens
    repetition_penalty=1.15,  # Penalizes repeating tokens
    eos_token_id=tokenizer.eos_token_id, # Stop when EOS is generated
    pad_token_id=tokenizer.pad_token_id, # Set pad token
)

print(f"\nPROMPT:\n{prompt}")

with torch.no_grad():
    out = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        **gen_kwargs
    )

# Decode the generated tokens, skipping the prompt
gen_tokens = out[0][enc["input_ids"].shape[-1]:]
reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

print(f"\nMODEL REPLY:\n{reply}")
USE_LORA = True

print(f"--- Settings ---")
print(f"MODEL_ID: {MODEL_ID}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"---")

# --- 2. Device Setup ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if torch.cuda.is_available():
    device = 'cuda'
elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

model_dtype = torch.float32 
use_pin_memory = True if device == 'cuda' else False

print(f'Using device: {device} (model dtype={model_dtype})')

# --- 3. Load Tokenizer (Moved Early) ---
# We need the tokenizer loaded so we can access its eos_token
# for the dataset formatting helper.
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) 
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

# --- UPDATED: Get the EOS token ---
eos_token = tokenizer.eos_token
print(f"Tokenizer loaded. EOS token: {eos_token}")


# --- 4. Dataset Helpers (Updated) ---
def safe_load_customer_dataset(dataset_id):
    try:
        ds = load_dataset(dataset_id)
        return ds.get('train', ds)
    except Exception as e:
        print(f'Could not load dataset {dataset_id}: {e}')
        print('Falling back to a tiny synthetic customer support dataset.')
        samples = [
            {'customer': "My order hasn't arrived, it's been 10 days.", 'agent': "I'm sorry. Can you share your order id?"},
            {'customer': 'I was charged twice for the same order.', 'agent': "I can help. Please share the transaction id."},
        ]
        return Dataset.from_list(samples)

def build_prompt_for_training(row):
    # --- UPDATED: Added {eos_token} to teach the model when to stop ---
    if 'customer' in row and 'agent' in row:
        return {'text': f"Human: {row['customer']}\nAssistant: {row['agent']}{eos_token}\n"}
    if 'input' in row and 'output' in row:
        return {'text': f"Human: {row['input']}\nAssistant: {row['output']}{eos_token}\n"}
    if 'instruction' in row and 'response' in row:
         return {'text': f"Human: {row['instruction']}\nAssistant: {row['response']}{eos_token}\n"}
    return {'text': str(row)}

# --- 5. Load and Prepare Dataset (Was Step 4) ---
print("Loading dataset...")
raw_ds = safe_load_customer_dataset(DATASET_ID)
ds = raw_ds.map(build_prompt_for_training)

# Split and reduce for a local run
if len(ds) > 2000:
    ds_split = ds.train_test_split(test_size=0.05, shuffle=True, seed=42)
    train_ds = ds_split['train'].select(range(min(4096, len(ds_split['train']))))
    eval_ds = ds_split['test'].select(range(min(128, len(ds_split['test']))))
else:
    ds_split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = ds_split['train']
    eval_ds = ds_split['test']

print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

# --- 6. Load Model (Was part of Step 5) ---
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=False,
    torch_dtype=model_dtype,
    low_cpu_mem_usage=True,
    attn_implementation="eager"
)
# Resize embeddings *before* PEFT
model.resize_token_embeddings(len(tokenizer))
print("Base model loaded.")

# --- 7. Apply LoRA/PEFT (Was Step 6) ---
if USE_LORA:
    print("Applying LoRA adapter (PEFT)...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"], # Correct for distilgpt2
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("LoRA adapter applied.")
else:
    print("Skipping LoRA. Training full model (slower).")

# --- 8. Tokenize Dataset (Was Step 7) ---
def tokenize_for_lm(examples):
    texts = [str(t) for t in examples.get('text', [])]
    outputs = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

print("Tokenizing datasets...")
train_tok = train_ds.map(tokenize_for_lm, batched=True, remove_columns=train_ds.column_names)
eval_tok = eval_ds.map(tokenize_for_lm, batched=True, remove_columns=eval_ds.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- 9. Define Trainer (Was Step 8) ---
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=10,
    save_total_limit=2,
    fp16=False, 
    remove_unused_columns=False,
    dataloader_pin_memory=use_pin_memory,
    use_mps_device=(device == 'mps'), 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=data_collator,
)

# --- 10. !! RUN TRAINING AND SAVE MODEL !! (Was Step 9) ---
print("--- Starting Training ---")
trainer.train()
print("--- Training Complete ---")

print(f"Saving fine-tuned adapter to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Adapter saved successfully.")


# --- 11. Test the Fine-Tuned Model (Was Step 10) ---
print("\n--- Testing the Fine-Tuned Model ---")

# Clear memory
del model
del trainer
if device == 'cuda':
    torch.cuda.empty_cache()
elif device == 'mps':
    torch.mps.empty_cache()

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=False,
    torch_dtype=model_dtype,
    attn_implementation="eager"
)
# 2. Load the tokenizer from our saved directory
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

# Resize the base model embeddings *before* loading the adapter
base_model.resize_token_embeddings(len(tokenizer))

# 3. Apply the adapter
model = PeftModel.from_pretrained(
    base_model,
    OUTPUT_DIR,
    torch_dtype=model_dtype,
)
model = model.to(device)
model.eval()

print("Fine-tuned model reloaded for testing.")

# Test generation
prompt = "Human: I haven't received my refund after 10 days. What should I do?\nAssistant:"
enc = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device)

with torch.no_grad():
    # --- UPDATED: Added repetition_penalty and changed temperature ---
    gen_kwargs = dict(
        max_new_tokens=80,
        do_sample=True,
        temperature=0.5,        # Was 0.22
        top_k=40,
        repetition_penalty=1.15,  # Added this to discourage loops
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    out = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        **gen_kwargs
    )

gen_tokens = out[0][enc["input_ids"].shape[-1]:]
reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

print(f"\nPROMPT:\n{prompt}")
print(f"\nMODEL REPLY:\n{reply}")