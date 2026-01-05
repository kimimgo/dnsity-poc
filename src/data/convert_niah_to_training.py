"""
Convert NIAH evaluation dataset to Gist Token training format.

Adds instruction and gist_tokens fields required for training.
"""

import json
from pathlib import Path
from typing import List, Dict
import argparse


def convert_sample(sample: Dict, num_gist_tokens: int = 10) -> Dict:
    """
    Convert NIAH sample to training format.

    Args:
        sample: NIAH sample with context, question, answer
        num_gist_tokens: Number of Gist tokens to use

    Returns:
        Training sample with instruction and gist_tokens added
    """
    # Generate Gist token string
    gist_tokens = "".join([f"<GIST_{i}>" for i in range(num_gist_tokens)])

    # Create instruction
    instruction = (
        "Read the following text carefully and compress its key information "
        "into the Gist tokens provided. Then answer the question based solely "
        "on the information stored in the Gist tokens."
    )

    return {
        "instruction": instruction,
        "context": sample["context"],
        "gist_tokens": gist_tokens,
        "question": sample["question"],
        "answer": sample["answer"]
    }


def convert_dataset(
    input_path: Path,
    output_path: Path,
    num_gist_tokens: int = 10,
    limit: int = None
):
    """
    Convert entire NIAH dataset to training format.

    Args:
        input_path: Path to NIAH JSONL file
        output_path: Path to save training JSONL file
        num_gist_tokens: Number of Gist tokens
        limit: Optional limit on number of samples
    """
    samples = []

    # Load NIAH samples
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            sample = json.loads(line)
            training_sample = convert_sample(sample, num_gist_tokens)
            samples.append(training_sample)

    # Save training samples
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"✅ Converted {len(samples)} samples")
    print(f"📁 Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert NIAH to training format")
    parser.add_argument("--input", type=str, required=True, help="Input NIAH JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output training JSONL file")
    parser.add_argument("--num-gist-tokens", type=int, default=10, help="Number of Gist tokens")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")

    args = parser.parse_args()

    convert_dataset(
        Path(args.input),
        Path(args.output),
        num_gist_tokens=args.num_gist_tokens,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
