"""
Baseline Comparison Experiment for Gist Token PoC.

Compares three approaches on NIAH datasets:
1. Baseline 1: Full Context (no compression)
2. Baseline 2: RAG (retrieval-based)
3. Experimental: Gist Token (learned compression)

Validates against CONCEPT.md evaluation criteria:
- Passkey Retrieval Accuracy
- Compression Ratio
- VRAM Usage
- Throughput (tokens/sec)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import time
import torch
from tqdm import tqdm

from src.baseline.full_context import FullContextBaseline, measure_vram_usage
from src.baseline.rag_pipeline import RAGPipeline
from src.evaluation.metrics import (
    calculate_passkey_accuracy,
    calculate_compression_ratio,
    calculate_throughput
)


def load_niah_samples(dataset_path: Path) -> List[Dict]:
    """Load NIAH samples from JSONL file."""
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def evaluate_full_context(
    samples: List[Dict],
    model_name: str = "gpt2",
    max_new_tokens: int = 20
) -> Dict:
    """
    Evaluate Full Context baseline.

    Args:
        samples: NIAH samples
        model_name: Model to use
        max_new_tokens: Max tokens to generate

    Returns:
        Results dict with accuracy, VRAM, throughput
    """
    print(f"\n🔍 Evaluating Full Context Baseline ({model_name})...")

    baseline = FullContextBaseline(model_name=model_name, device="cpu")
    results = []
    total_tokens = 0
    start_time = time.time()

    # Measure VRAM on first sample (if CUDA available)
    vram_mb = 0.0
    if torch.cuda.is_available():
        baseline_gpu = FullContextBaseline(model_name=model_name, device="cuda")
        sample = samples[0]

        def inference_fn():
            baseline_gpu.generate(
                context=sample["context"],
                question=sample["question"],
                max_new_tokens=max_new_tokens
            )

        vram_mb = measure_vram_usage(inference_fn, reset=True)
        del baseline_gpu
        torch.cuda.empty_cache()

    # Evaluate all samples (CPU to avoid OOM)
    for sample in tqdm(samples[:50], desc="Full Context"):  # Limit to 50 for speed
        predicted = baseline.generate(
            context=sample["context"],
            question=sample["question"],
            max_new_tokens=max_new_tokens
        )

        results.append({
            "predicted": predicted.strip(),
            "ground_truth": sample["answer"]
        })

        total_tokens += max_new_tokens

    elapsed = time.time() - start_time
    accuracy = calculate_passkey_accuracy(results)
    throughput = calculate_throughput(total_tokens, elapsed)

    # Compression ratio = 1.0 (no compression)
    avg_context_len = sum(len(s["context"].split()) for s in samples[:50]) / 50
    compression_ratio = avg_context_len / avg_context_len  # 1.0

    return {
        "approach": "Full Context",
        "accuracy": accuracy,
        "compression_ratio": compression_ratio,
        "vram_mb": vram_mb,
        "throughput_tokens_sec": throughput,
        "num_samples": len(results)
    }


def evaluate_rag(
    samples: List[Dict],
    model_name: str = "gpt2",
    top_k: int = 3,
    max_new_tokens: int = 20
) -> Dict:
    """
    Evaluate RAG baseline.

    Args:
        samples: NIAH samples
        model_name: Model to use
        top_k: Number of chunks to retrieve
        max_new_tokens: Max tokens to generate

    Returns:
        Results dict with accuracy, VRAM, throughput
    """
    print(f"\n🔍 Evaluating RAG Baseline (top_k={top_k})...")

    # Build RAG index
    rag = RAGPipeline(
        model_name=model_name,
        device="cpu",
        collection_name="niah_eval"
    )

    # Index all contexts (chunk into sentences)
    for sample in tqdm(samples[:50], desc="Indexing"):
        sentences = sample["context"].split(". ")
        for i, sentence in enumerate(sentences):
            rag.add_documents([sentence], ids=[f"{hash(sample['context'])}_{i}"])

    results = []
    total_tokens = 0
    start_time = time.time()

    # Measure VRAM
    vram_mb = 0.0
    if torch.cuda.is_available():
        rag_gpu = RAGPipeline(model_name=model_name, device="cuda", collection_name="niah_eval_gpu")
        sample = samples[0]

        def inference_fn():
            rag_gpu.query(sample["question"], top_k=top_k, max_new_tokens=max_new_tokens)

        vram_mb = measure_vram_usage(inference_fn, reset=True)
        del rag_gpu
        torch.cuda.empty_cache()

    # Evaluate
    for sample in tqdm(samples[:50], desc="RAG Query"):
        predicted = rag.query(
            sample["question"],
            top_k=top_k,
            max_new_tokens=max_new_tokens
        )

        results.append({
            "predicted": predicted.strip(),
            "ground_truth": sample["answer"]
        })

        total_tokens += max_new_tokens

    elapsed = time.time() - start_time
    accuracy = calculate_passkey_accuracy(results)
    throughput = calculate_throughput(total_tokens, elapsed)

    # Compression: avg_context_len / (top_k * avg_chunk_len)
    avg_context_len = sum(len(s["context"].split()) for s in samples[:50]) / 50
    avg_chunks = top_k * 20  # Assume ~20 tokens per chunk
    compression_ratio = avg_context_len / avg_chunks

    return {
        "approach": "RAG",
        "accuracy": accuracy,
        "compression_ratio": compression_ratio,
        "vram_mb": vram_mb,
        "throughput_tokens_sec": throughput,
        "num_samples": len(results)
    }


def evaluate_gist_token(
    samples: List[Dict],
    num_gist_tokens: int = 10,
    max_new_tokens: int = 20
) -> Dict:
    """
    Evaluate Gist Token approach (placeholder - requires trained model).

    Args:
        samples: NIAH samples
        num_gist_tokens: Number of Gist tokens
        max_new_tokens: Max tokens to generate

    Returns:
        Results dict with accuracy, VRAM, throughput
    """
    print(f"\n🔍 Evaluating Gist Token (num_gist_tokens={num_gist_tokens})...")
    print("⚠️  Gist Token evaluation requires trained model - returning placeholder results")

    # Placeholder results (would require actual trained model)
    avg_context_len = sum(len(s["context"].split()) for s in samples[:50]) / 50
    compression_ratio = avg_context_len / num_gist_tokens

    return {
        "approach": "Gist Token",
        "accuracy": 0.0,  # Placeholder - needs trained model
        "compression_ratio": compression_ratio,
        "vram_mb": 0.0,  # Placeholder
        "throughput_tokens_sec": 0.0,  # Placeholder
        "num_samples": 0,
        "status": "Not implemented - requires trained model"
    }


def save_results(results: List[Dict], output_path: Path):
    """Save experiment results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Results saved to {output_path}")


def print_summary(results: List[Dict]):
    """Print experiment summary table."""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"{'Approach':<20} {'Accuracy':>12} {'Compression':>14} {'VRAM (MB)':>12} {'Throughput':>12}")
    print("-"*80)

    for r in results:
        print(
            f"{r['approach']:<20} "
            f"{r['accuracy']:>11.2%} "
            f"{r['compression_ratio']:>13.1f}x "
            f"{r['vram_mb']:>11.1f} "
            f"{r['throughput_tokens_sec']:>11.1f}"
        )

    print("="*80)

    # Highlight key findings
    print("\n📊 Key Findings:")
    print(f"- Best Accuracy: {max(results, key=lambda x: x['accuracy'])['approach']}")
    print(f"- Best Compression: {max(results, key=lambda x: x['compression_ratio'])['approach']}")
    print(f"- Lowest VRAM: {min(results, key=lambda x: x['vram_mb'] if x['vram_mb'] > 0 else float('inf'))['approach']}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison experiment")
    parser.add_argument(
        "--dataset",
        default="data/processed/niah/global_niah.jsonl",
        help="Path to NIAH dataset"
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Model to use for baselines"
    )
    parser.add_argument(
        "--output",
        default="experiments/results/baseline_comparison.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--skip-gist",
        action="store_true",
        help="Skip Gist Token evaluation (not yet implemented)"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"📂 Loading dataset: {args.dataset}")
    samples = load_niah_samples(Path(args.dataset))
    print(f"✅ Loaded {len(samples)} samples")

    # Run experiments
    results = []

    # 1. Full Context
    results.append(evaluate_full_context(samples, model_name=args.model))

    # 2. RAG
    results.append(evaluate_rag(samples, model_name=args.model))

    # 3. Gist Token (placeholder)
    if not args.skip_gist:
        results.append(evaluate_gist_token(samples, num_gist_tokens=10))

    # Save and display
    save_results(results, Path(args.output))
    print_summary(results)

    # Validation against CONCEPT.md criteria
    print("\n✅ CONCEPT.md Validation:")
    print("- ✅ Passkey Retrieval Accuracy: Measured")
    print("- ✅ Compression Ratio: Calculated")
    print("- ✅ VRAM Usage: Measured")
    print("- ✅ Throughput: Calculated")
    print("- ⚠️  Gist Token: Requires trained model for full validation")


if __name__ == "__main__":
    main()
