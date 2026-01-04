"""
CPU-only baseline experiment for proof-of-concept validation.

Simplified version that runs on CPU to validate the entire pipeline.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import time
from tqdm import tqdm

# Simplified CPU-only baseline
class SimpleCPUBaseline:
    def __init__(self):
        pass

    def evaluate_sample(self, sample: Dict) -> Dict:
        """Simulate evaluation (passkey extraction from context)."""
        context = sample["context"]
        answer = sample["answer"]

        # Simple heuristic: search for passkey pattern in context
        import re
        # English pattern
        match = re.search(r'passkey is (\w+)', context, re.IGNORECASE)
        if not match:
            # Korean pattern
            match = re.search(r'비밀 패스키는 (\w+)', context)

        if match:
            predicted = match.group(1)
            # Remove Korean suffix if present (e.g., "ABC123입니다" -> "ABC123")
            predicted = re.sub(r'[가-힣]+$', '', predicted)
        else:
            predicted = "UNKNOWN"

        return {
            "predicted": predicted,
            "ground_truth": answer,
            "correct": predicted == answer
        }


def load_dataset(path: Path) -> List[Dict]:
    """Load NIAH dataset."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def calculate_accuracy(results: List[Dict]) -> float:
    """Calculate accuracy."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["correct"])
    return correct / len(results)


def run_experiment(dataset_path: str, output_path: str, limit: int = 50):
    """Run CPU baseline experiment."""
    print(f"\n📂 Loading dataset: {dataset_path}")
    samples = load_dataset(Path(dataset_path))
    print(f"✅ Loaded {len(samples)} samples")

    # Limit samples for speed
    samples = samples[:limit]
    print(f"🔬 Running experiment on {len(samples)} samples...")

    baseline = SimpleCPUBaseline()
    results = []

    start_time = time.time()

    for sample in tqdm(samples, desc="Evaluating"):
        result = baseline.evaluate_sample(sample)
        results.append(result)

    elapsed = time.time() - start_time
    accuracy = calculate_accuracy(results)

    # Calculate stats
    avg_context_len = sum(len(s["context"]) for s in samples) / len(samples)

    # Summary
    summary = {
        "dataset": dataset_path,
        "num_samples": len(samples),
        "accuracy": accuracy,
        "elapsed_time_sec": elapsed,
        "avg_context_length_chars": avg_context_len,
        "results": results
    }

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT RESULTS")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_path}")
    print(f"Samples: {len(samples)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Elapsed Time: {elapsed:.1f}s")
    print(f"Avg Context Length: {avg_context_len:.0f} chars")
    print(f"{'='*80}")
    print(f"\n📁 Results saved to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=50)

    args = parser.parse_args()

    run_experiment(args.dataset, args.output, args.limit)


if __name__ == "__main__":
    main()
