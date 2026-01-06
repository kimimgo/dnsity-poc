#!/usr/bin/env python3
"""KMMLU evaluation script using vLLM OpenAI-compatible API."""
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm


def format_question(question: str, choices: list[str]) -> str:
    """Format KMMLU question with choices."""
    prompt = f"""다음 문제를 읽고 정답을 선택하세요. 정답만 A, B, C, D 중 하나로 답하세요.

문제: {question}

"""
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        prompt += f"{letter}. {choice}\n"

    prompt += "\n정답:"
    return prompt


def extract_answer(response: str) -> str:
    """Extract answer letter from model response."""
    response = response.strip().upper()

    # Direct letter match at start
    if response and response[0] in 'ABCD':
        return response[0]

    # Pattern: "정답: X" or "답: X"
    match = re.search(r'(?:정답|답)[:\s]*([ABCD])', response)
    if match:
        return match.group(1)

    # Any letter in response
    for char in response:
        if char in 'ABCD':
            return char

    return 'X'  # Unknown


def evaluate_kmmlu(
    client: OpenAI,
    model: str,
    subject: str = "Law",
    num_samples: int = 100,
    output_dir: str = "results/kmmlu"
):
    """Run KMMLU evaluation."""
    print(f"Loading KMMLU dataset (subject: {subject})...")

    # Load dataset - KMMLU uses subject names like "Law", "Criminal Law", etc.
    ds = load_dataset("HAERAE-HUB/KMMLU", subject, split="test")

    # Limit samples
    samples = list(ds)[:num_samples]
    print(f"Evaluating {len(samples)} samples...")

    results = []
    correct = 0

    for item in tqdm(samples, desc="Evaluating"):
        question = item["question"]
        choices = [item["A"], item["B"], item["C"], item["D"]]
        answer_idx = item["answer"]  # 1-indexed: 1=A, 2=B, 3=C, 4=D
        ground_truth = chr(ord('A') + answer_idx - 1)

        prompt = format_question(question, choices)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,  # Reasoning models need more tokens
                temperature=0.0,
            )
            message = response.choices[0].message
            # gpt-oss uses reasoning mode - content may be null during reasoning
            generated = message.content
            if generated is None:
                # Fallback to reasoning_content if available
                generated = getattr(message, 'reasoning_content', '') or ''
            predicted = extract_answer(generated)
        except Exception as e:
            print(f"Error: {e}")
            generated = ""
            predicted = "X"

        is_correct = (predicted == ground_truth)
        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "choices": choices,
            "ground_truth": ground_truth,
            "ground_truth_idx": answer_idx,
            "predicted": predicted,
            "correct": is_correct,
            "generated": generated,
        })

    accuracy = (correct / len(samples)) * 100

    output = {
        "model": model,
        "subject": subject,
        "num_samples": len(samples),
        "accuracy": accuracy,
        "correct": correct,
        "total": len(samples),
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"gptoss20b_{subject.lower().replace(' ', '_')}_{num_samples}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print(f"Subject: {subject}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(samples)})")
    print(f"Results saved to: {output_file}")
    print(f"{'='*50}")

    return output


def main():
    parser = argparse.ArgumentParser(description="KMMLU evaluation with vLLM")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name")
    parser.add_argument("--subject", default="Law", help="KMMLU subject")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--output-dir", default="results/kmmlu", help="Output directory")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    evaluate_kmmlu(
        client=client,
        model=args.model,
        subject=args.subject,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
