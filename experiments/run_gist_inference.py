"""
Run inference with trained Gist Token model on NIAH dataset.

Measures Passkey Retrieval Accuracy, Compression Ratio, and VRAM Usage.
"""

import os
import json
import torch
import argparse
import re
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


def load_niah_dataset(file_path: Path, limit: int = None) -> List[Dict]:
    """Load NIAH dataset."""
    samples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            samples.append(json.loads(line))
    return samples


def load_gist_model(checkpoint_dir: Path, device: str = "cuda"):
    """Load trained Gist Token model from checkpoint with 4-bit quantization."""
    print(f"📥 Loading model from {checkpoint_dir}")

    # Load tokenizer (includes gist tokens)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    print(f"✅ Loaded tokenizer with {len(tokenizer)} tokens")

    # Load base model with 4-bit quantization (same as training)
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    # CRITICAL: Resize embeddings to match checkpoint (128256 -> 128266 with 10 gist tokens)
    model.resize_token_embeddings(len(tokenizer))
    print(f"✅ Resized embeddings to {len(tokenizer)} tokens")

    # Load LoRA weights (keep them as adapters, don't merge)
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.config.use_cache = True

    print(f"✅ Model loaded on {device} with 4-bit quantization")
    return model, tokenizer


def extract_passkey(text: str) -> str:
    """Extract passkey from generated text."""
    # English pattern
    match = re.search(r'passkey is (\w+)', text, re.IGNORECASE)
    if match:
        return match.group(1)

    # Korean pattern
    match = re.search(r'비밀 패스키는 (\w+)', text)
    if match:
        predicted = match.group(1)
        # Remove Korean suffix
        predicted = re.sub(r'[가-힣]+$', '', predicted)
        return predicted

    # Answer pattern
    match = re.search(r'Answer: (\w+)', text)
    if match:
        return match.group(1)

    # Last word pattern
    words = text.strip().split()
    if words:
        last_word = re.sub(r'[^\w]', '', words[-1])
        if last_word:
            return last_word

    return "UNKNOWN"


def run_inference(model, tokenizer, sample: Dict, max_new_tokens: int = 50) -> Dict:
    """Run inference on a single NIAH sample."""
    # Create prompt
    prompt = f"Read the following text and answer the question.\n\n"
    prompt += f"Context: {sample['context']}\n\n"
    prompt += f"Question: {sample['question']}\n\n"
    prompt += f"Answer:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer
    answer_part = generated_text.split("Answer:")[-1].strip()
    predicted = extract_passkey(answer_part)

    return {
        "predicted": predicted,
        "ground_truth": sample["answer"],
        "correct": predicted == sample["answer"],
        "generated_text": answer_part[:200]  # First 200 chars
    }


def measure_vram():
    """Measure peak VRAM usage."""
    if torch.cuda.is_available():
        vram_bytes = torch.cuda.max_memory_allocated()
        vram_gb = vram_bytes / (1024 ** 3)
        return vram_gb
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Run Gist Token inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint directory")
    parser.add_argument("--dataset", type=str, required=True, help="NIAH dataset JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output results JSON")
    parser.add_argument("--limit", type=int, default=50, help="Limit number of samples")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    print("=" * 80)
    print("🔬 Gist Token Inference")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print("=" * 80 + "\n")

    # Load model
    model, tokenizer = load_gist_model(Path(args.checkpoint), args.device)

    # Load dataset
    print(f"📂 Loading dataset from {args.dataset}")
    samples = load_niah_dataset(Path(args.dataset), args.limit)
    print(f"✅ Loaded {len(samples)} samples\n")

    # Reset VRAM stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Run inference
    print("🧪 Running inference...")
    results = []
    correct = 0

    for sample in tqdm(samples, desc="Inference"):
        result = run_inference(model, tokenizer, sample)
        results.append(result)
        if result["correct"]:
            correct += 1

    # Calculate metrics
    accuracy = correct / len(results) if results else 0.0
    vram_gb = measure_vram()

    # Calculate compression ratio
    avg_context_tokens = sum(len(tokenizer.encode(s["context"])) for s in samples) / len(samples)
    num_gist_tokens = 10  # From checkpoint config
    compression_ratio = avg_context_tokens / num_gist_tokens

    # Summary
    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "num_samples": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "vram_gb": vram_gb,
        "avg_context_tokens": avg_context_tokens,
        "num_gist_tokens": num_gist_tokens,
        "compression_ratio": compression_ratio,
        "results": results[:10]  # First 10 for inspection
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("📊 INFERENCE RESULTS")
    print("=" * 80)
    print(f"Samples: {len(results)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"VRAM Usage: {vram_gb:.2f} GB")
    print(f"Avg Context Tokens: {avg_context_tokens:.0f}")
    print(f"Gist Tokens: {num_gist_tokens}")
    print(f"Compression Ratio: {compression_ratio:.1f}x")
    print("=" * 80)
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
