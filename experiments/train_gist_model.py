"""
Train Llama-3-8B with Gist Tokens.

Complete training script with model loading, data preparation, and training.
"""

import os
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset


def load_training_data(file_path: Path) -> List[Dict]:
    """Load training data from JSONL file."""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def add_gist_tokens(tokenizer, num_gist_tokens: int = 10):
    """Add Gist special tokens to tokenizer."""
    gist_tokens = [f"<GIST_{i}>" for i in range(num_gist_tokens)]
    tokenizer.add_special_tokens({'additional_special_tokens': gist_tokens})
    print(f"✅ Added {num_gist_tokens} Gist tokens")
    return tokenizer


def prepare_model(model_name: str, tokenizer, num_gist_tokens: int = 10):
    """Load and prepare model with LoRA and Gist tokens."""
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    print(f"📥 Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # Resize embeddings for Gist tokens
    print(f"🔧 Resizing embeddings for {num_gist_tokens} Gist tokens")
    model.resize_token_embeddings(len(tokenizer))

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"]  # CRITICAL for Gist tokens
    )

    print("🔧 Applying LoRA")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def format_sample(sample: Dict, tokenizer) -> str:
    """
    Format training sample as text prompt.

    Format: [Instruction] [Context] [Gist Tokens] [Question] [Answer]
    """
    text = f"{sample['instruction']}\n\n"
    text += f"Context: {sample['context']}\n\n"
    text += f"Gist Tokens: {sample['gist_tokens']}\n\n"
    text += f"Question: {sample['question']}\n\n"
    text += f"Answer: {sample['answer']}"

    return text


def tokenize_function(sample: Dict, tokenizer, max_length: int = 2048):
    """Tokenize a single sample with truncation."""
    # Format as text
    text = format_sample(sample, tokenizer)

    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    # Labels are same as input_ids for causal LM
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Train Gist Token model")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--train-data", type=str, required=True, help="Training JSONL file")
    parser.add_argument("--output-dir", type=str, default="checkpoints/gist-10")
    parser.add_argument("--num-gist-tokens", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 Gist Token Training - Starting")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Training Data: {args.train_data}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Num Gist Tokens: {args.num_gist_tokens}")
    print(f"Max Steps: {args.max_steps}")
    print("=" * 80)

    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Add Gist tokens
    tokenizer = add_gist_tokens(tokenizer, args.num_gist_tokens)

    # Load model
    model = prepare_model(args.model, tokenizer, args.num_gist_tokens)

    # Load training data
    print(f"\n📂 Loading training data from {args.train_data}...")
    train_samples = load_training_data(Path(args.train_data))
    print(f"✅ Loaded {len(train_samples)} training samples")

    # Create Dataset - directly from list of dicts
    train_dataset = Dataset.from_list(train_samples)

    # Tokenize
    print("\n🔤 Tokenizing samples...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=False,
        remove_columns=train_dataset.column_names
    )
    print(f"✅ Tokenized {len(train_dataset)} samples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none"
    )

    # Create Trainer
    print("\n🏋️ Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    # Train
    print("\n" + "=" * 80)
    print("🏃 TRAINING STARTED")
    print("=" * 80)
    print(f"Expected duration: ~{args.max_steps * 20 / 3600:.1f} hours")
    print("=" * 80 + "\n")

    trainer.train()

    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED")
    print("=" * 80)
    print(f"Model saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
