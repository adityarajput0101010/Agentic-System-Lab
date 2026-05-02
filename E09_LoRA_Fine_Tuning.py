# ============================================================
# DKTE Textile & Engineering Institute, Ichalkaranji
# Department: CSE (AI & ML)
# Subject: 01AMP347 - Agentic Systems Lab
# Experiment No: E09
# Title: Fine-Tuning LLMs using Low-Rank Adaptation (LoRA)
# ============================================================
# Install: pip install transformers peft datasets accelerate trl
#          pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# NOTE: GPU recommended for faster training. CPU training is slow
#       but works for demonstration purposes.

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from trl import SFTTrainer

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
MODEL_NAME   = "gpt2"              # Small model for demonstration
OUTPUT_DIR   = "./lora_finetuned"
LORA_RANK    = 8                   # Rank of LoRA matrices (r)
LORA_ALPHA   = 16                  # Scaling factor
LORA_DROPOUT = 0.1
TARGET_MODULES = ["c_attn"]        # GPT-2 attention layers

TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,                    # Set True if GPU with float16 support
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    warmup_steps=10,
    report_to="none",
)


# ─────────────────────────────────────────────
#  SAMPLE TRAINING DATASET
# ─────────────────────────────────────────────
TRAINING_DATA = [
    {"text": "Q: What is machine learning?\nA: Machine learning is a subset of AI that enables systems to learn from data and improve without explicit programming."},
    {"text": "Q: What is a neural network?\nA: A neural network is a computational model inspired by the human brain, consisting of layers of interconnected nodes called neurons."},
    {"text": "Q: What is deep learning?\nA: Deep learning uses neural networks with many layers to learn complex patterns from large datasets."},
    {"text": "Q: What is NLP?\nA: Natural Language Processing (NLP) is a branch of AI that enables computers to understand, interpret, and generate human language."},
    {"text": "Q: What is a transformer?\nA: A transformer is a neural network architecture based on self-attention mechanisms, widely used in NLP tasks."},
    {"text": "Q: What is transfer learning?\nA: Transfer learning is using a pre-trained model as a starting point for a new task, saving time and resources."},
    {"text": "Q: What is overfitting?\nA: Overfitting occurs when a model learns training data too well, including noise, leading to poor generalization on new data."},
    {"text": "Q: What is regularization?\nA: Regularization techniques like L1/L2 or dropout prevent overfitting by adding constraints or noise during training."},
    {"text": "Q: What is gradient descent?\nA: Gradient descent is an optimization algorithm that minimizes the loss function by updating model parameters in the direction of the negative gradient."},
    {"text": "Q: What is backpropagation?\nA: Backpropagation is the algorithm used to compute gradients of the loss function with respect to model weights, enabling gradient descent."},
    {"text": "Q: What is an embedding?\nA: An embedding is a dense vector representation of discrete objects (words, categories) in a continuous vector space."},
    {"text": "Q: What is attention mechanism?\nA: The attention mechanism allows models to focus on relevant parts of the input when producing each part of the output."},
    {"text": "Q: What is fine-tuning?\nA: Fine-tuning adapts a pre-trained model to a specific task by training it on a smaller task-specific dataset."},
    {"text": "Q: What is LoRA?\nA: LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that injects trainable low-rank matrices into transformer layers."},
    {"text": "Q: What is RAG?\nA: RAG (Retrieval-Augmented Generation) combines information retrieval with text generation for accurate, grounded responses."},
]


# ─────────────────────────────────────────────
#  STEP 1: LOAD MODEL & TOKENIZER
# ─────────────────────────────────────────────
def load_model_and_tokenizer():
    print("\n" + "="*60)
    print("STEP 1: Loading Base Model and Tokenizer")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer loaded: {MODEL_NAME}")
    print(f"   Vocab size: {tokenizer.vocab_size}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    print(f"✅ Model loaded: {MODEL_NAME}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    return model, tokenizer


# ─────────────────────────────────────────────
#  STEP 2: APPLY LoRA
# ─────────────────────────────────────────────
def apply_lora(model):
    print("\n" + "="*60)
    print("STEP 2: Applying LoRA Configuration")
    print("="*60)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    print(f"   LoRA rank (r)     : {LORA_RANK}")
    print(f"   LoRA alpha        : {LORA_ALPHA}")
    print(f"   Target modules    : {TARGET_MODULES}")
    print(f"   Dropout           : {LORA_DROPOUT}")

    peft_model = get_peft_model(model, lora_config)

    trainable     = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total         = sum(p.numel() for p in peft_model.parameters())
    print(f"\n✅ LoRA applied successfully!")
    print(f"   Trainable parameters : {trainable:,}")
    print(f"   Total parameters     : {total:,}")
    print(f"   Trainable %          : {100 * trainable / total:.4f}%")

    return peft_model


# ─────────────────────────────────────────────
#  STEP 3: PREPARE DATASET
# ─────────────────────────────────────────────
def prepare_dataset(tokenizer):
    print("\n" + "="*60)
    print("STEP 3: Preparing Training Dataset")
    print("="*60)

    dataset = Dataset.from_list(TRAINING_DATA)
    print(f"✅ Dataset created with {len(dataset)} examples")
    print(f"   Sample: {dataset[0]['text'][:80]}...")
    return dataset


# ─────────────────────────────────────────────
#  STEP 4: TRAIN
# ─────────────────────────────────────────────
def train_model(model, tokenizer, dataset):
    print("\n" + "="*60)
    print("STEP 4: Training with LoRA")
    print("="*60)

    trainer = SFTTrainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=256,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training complete!")

    # Save the adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ LoRA adapter saved to: {OUTPUT_DIR}")

    return trainer


# ─────────────────────────────────────────────
#  STEP 5: INFERENCE
# ─────────────────────────────────────────────
def inference(model, tokenizer, prompt):
    print(f"\n📝 Prompt: {prompt}")
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"💬 Response: {response}")
    return response


# ─────────────────────────────────────────────
#  STEP 6: LoRA MATH EXPLANATION
# ─────────────────────────────────────────────
def explain_lora_math():
    print("\n" + "="*60)
    print("STEP 6: LoRA Mathematics Explained")
    print("="*60)
    print("""
  LoRA modifies weight matrices as: W' = W + ΔW = W + (A × B)
  
  Where:
    W  ∈ R^(d × k)  — Original frozen weight matrix
    A  ∈ R^(d × r)  — Trainable low-rank matrix (initialized random)
    B  ∈ R^(r × k)  — Trainable low-rank matrix (initialized zeros)
    r               — Rank (r << min(d, k))
  
  Example: d=768, k=768, r=8
    Original params : 768 × 768 = 589,824
    LoRA params     : 768×8 + 8×768 = 12,288
    Reduction       : 98% fewer trainable parameters!
  """)

    import math
    d, k, r = 768, 768, 8
    original = d * k
    lora     = (d * r) + (r * k)
    print(f"  Numerical Example (d={d}, k={k}, r={r}):")
    print(f"    Original parameters : {original:,}")
    print(f"    LoRA parameters     : {lora:,}")
    print(f"    Parameter reduction : {100*(1 - lora/original):.1f}%")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  E09: Fine-Tuning LLMs using LoRA")
    print("=" * 60)

    explain_lora_math()

    # Full training pipeline
    model, tokenizer = load_model_and_tokenizer()
    model            = apply_lora(model)
    dataset          = prepare_dataset(tokenizer)
    trainer          = train_model(model, tokenizer, dataset)

    # Test inference
    print("\n" + "="*60)
    print("STEP 5: Testing Fine-Tuned Model")
    print("="*60)
    test_prompts = [
        "Q: What is machine learning?\nA:",
        "Q: What is LoRA?\nA:",
        "Q: What is a transformer?\nA:",
    ]
    for prompt in test_prompts:
        inference(model, tokenizer, prompt)

    print("\n✅ Experiment E09 Completed Successfully!")
