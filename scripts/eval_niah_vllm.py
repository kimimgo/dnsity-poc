#!/usr/bin/env python3
"""NIAH (Needle In A Haystack) evaluation using vLLM OpenAI-compatible API."""
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm


def load_niah_dataset(filepath: str, limit: int = None) -> list[dict]:
    """Load NIAH dataset from JSONL file."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            samples.append(json.loads(line))
    return samples


def extract_passkey(response: str, needle: str) -> tuple[str, bool]:
    """Extract passkey from model response and check if correct."""
    if response is None:
        return "", False

    response_clean = response.strip().upper()
    needle_upper = needle.upper()

    # Direct match
    if needle_upper in response_clean:
        return needle, True

    # Pattern: 6-character alphanumeric (passkey format)
    patterns = re.findall(r'[A-Z0-9]{6}', response_clean)
    for p in patterns:
        if p == needle_upper:
            return needle, True

    # Return first 6-char pattern found, or raw response
    if patterns:
        return patterns[0], patterns[0] == needle_upper

    return response_clean[:20], False


def evaluate_niah(
    client: OpenAI,
    model: str,
    dataset_path: str,
    num_samples: int = 200,
    max_context_tokens: int = 2500,  # Reduced for reasoning models (need more output tokens)
    output_dir: str = "results"
):
    """Run NIAH evaluation."""
    print(f"Loading NIAH dataset from {dataset_path}...")
    samples = load_niah_dataset(dataset_path, limit=num_samples)
    print(f"Evaluating {len(samples)} samples...")

    results = []
    correct = 0

    for item in tqdm(samples, desc="Evaluating"):
        context = item["context"]
        question = item["question"]
        needle = item["answer"]

        # Truncate context if too long (rough estimate: 1 token ~ 3 chars for Korean/CJK)
        # Korean text is denser in tokens
        is_korean = any('\uac00' <= c <= '\ud7af' for c in context[:100])
        chars_per_token = 2 if is_korean else 4
        max_chars = max_context_tokens * chars_per_token

        if len(context) > max_chars:
            # Keep needle by finding its position
            needle_pos = context.find(needle)
            if needle_pos > 0:
                # Keep context around the needle, ensuring needle is included
                buffer = 200  # Extra buffer around needle
                half_window = (max_chars - buffer) // 2
                start = max(0, needle_pos - half_window)
                end = min(len(context), needle_pos + len(needle) + half_window)

                # Ensure we don't exceed max_chars
                if end - start > max_chars:
                    end = start + max_chars

                context = context[start:end]
            else:
                context = context[:max_chars]

        prompt = f"""Read the following text and answer the question.

Text:
{context}

Question: {question}

Answer with ONLY the passkey (6 characters). Do not include any explanation."""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,  # Reasoning models need more tokens
                temperature=0.0,
            )
            message = response.choices[0].message
            generated = message.content
            if generated is None:
                # Fallback to reasoning_content
                generated = getattr(message, 'reasoning_content', '') or ''

            predicted, is_correct = extract_passkey(generated, needle)
        except Exception as e:
            print(f"Error: {e}")
            generated = ""
            predicted = ""
            is_correct = False

        if is_correct:
            correct += 1

        results.append({
            "needle": needle,
            "predicted": predicted,
            "correct": is_correct,
            "needle_position": item.get("needle_position", 0),
            "context_length": len(item["context"]),
            "generated": generated[:200] if generated else "",
        })

    accuracy = (correct / len(samples)) * 100

    # Calculate accuracy by position bucket
    position_buckets = {"0-20%": [], "20-40%": [], "40-60%": [], "60-80%": [], "80-100%": []}
    for r in results:
        pos = r["needle_position"]
        if pos < 0.2:
            position_buckets["0-20%"].append(r["correct"])
        elif pos < 0.4:
            position_buckets["20-40%"].append(r["correct"])
        elif pos < 0.6:
            position_buckets["40-60%"].append(r["correct"])
        elif pos < 0.8:
            position_buckets["60-80%"].append(r["correct"])
        else:
            position_buckets["80-100%"].append(r["correct"])

    position_accuracy = {
        bucket: (sum(vals) / len(vals) * 100 if vals else 0)
        for bucket, vals in position_buckets.items()
    }

    dataset_name = Path(dataset_path).stem
    output = {
        "model": model,
        "dataset": dataset_name,
        "num_samples": len(samples),
        "accuracy": accuracy,
        "correct": correct,
        "total": len(samples),
        "position_accuracy": position_accuracy,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"gptoss20b_niah_{dataset_name}_{len(samples)}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Dataset: {dataset_name}")
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{len(samples)})")
    print(f"\nAccuracy by Needle Position:")
    for bucket, acc in position_accuracy.items():
        count = len(position_buckets[bucket])
        print(f"  {bucket}: {acc:.1f}% ({count} samples)")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}")

    return output


def main():
    parser = argparse.ArgumentParser(description="NIAH evaluation with vLLM")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name")
    parser.add_argument("--dataset", required=True, help="Path to NIAH dataset (JSONL)")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    evaluate_niah(
        client=client,
        model=args.model,
        dataset_path=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
