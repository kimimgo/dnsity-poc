"""
Gist Token 모델을 NoLiMa 데이터셋으로 평가 - 진실의 순간
기존 NIAH에서 34.5%였던 정확도가 NoLiMa에서는 얼마나 나올지 확인
"""
import json
import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from tqdm import tqdm

def load_model(checkpoint_dir):
    print(f"Loading model from {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True
    )

    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, checkpoint_dir)

    print("Model loaded")
    return model, tokenizer

def evaluate_sample(model, tokenizer, sample, num_gist_tokens=25):
    """Gist Token 평가 - 단순화된 버전"""
    # 컨텍스트 자르기
    max_ctx_len = 3500
    ctx_tokens = tokenizer.encode(sample["context"], add_special_tokens=False)
    if len(ctx_tokens) > max_ctx_len:
        ctx_tokens = ctx_tokens[:max_ctx_len]
    context = tokenizer.decode(ctx_tokens)

    # Gist 토큰 추가
    gist_str = "".join([f"<GIST_{i}>" for i in range(num_gist_tokens)])
    prompt = f"{context}\n\n{gist_str}\n\nQuestion: {sample['question']}\nAnswer:"

    # 생성
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # 6자리 코드 추출
    matches = re.findall(r'\b[A-Z0-9]{6}\b', generated.upper())
    predicted = matches[0] if matches else ""

    ground_truth = sample["answer"].upper()
    correct = predicted == ground_truth

    return {
        "predicted": predicted,
        "ground_truth": ground_truth,
        "correct": correct,
        "generated": generated[:100],
        "question": sample["question"],
        "needle_phrase": sample.get("needle_phrase", "")[:50]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/gist-25-1000")
    parser.add_argument("--dataset", default="data/nolima/nolima_200.jsonl")
    parser.add_argument("--num-gist-tokens", type=int, default=25)
    parser.add_argument("--output", default="results/gist_nolima.json")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    # 모델 로드
    model, tokenizer = load_model(args.checkpoint)

    # 데이터 로드
    print(f"Loading {args.dataset}")
    with open(args.dataset) as f:
        samples = [json.loads(line) for line in f]

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Evaluating {len(samples)} samples...")

    # 평가
    results = []
    for sample in tqdm(samples):
        try:
            result = evaluate_sample(model, tokenizer, sample, args.num_gist_tokens)
            results.append(result)
        except Exception as e:
            print(f"\nError: {e}")
            results.append({
                "predicted": "",
                "ground_truth": sample["answer"],
                "correct": False,
                "error": str(e)
            })

    # 결과 집계
    correct_count = sum(r["correct"] for r in results)
    accuracy = correct_count / len(results) * 100

    print("\n" + "=" * 70)
    print("GIST TOKEN - NoLiMa EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(results)})")
    print("=" * 70)

    # 오답 분석
    print("\nIncorrect samples (first 5):")
    incorrect = [r for r in results if not r["correct"]][:5]
    for i, r in enumerate(incorrect):
        print(f"  [{i+1}] Q: {r.get('question', '')[:60]}...")
        print(f"      Expected: {r['ground_truth']}, Got: {r['predicted']}")
        print(f"      Generated: {r.get('generated', '')[:60]}...")

    # 결과 저장
    output_data = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "num_gist_tokens": args.num_gist_tokens,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(results),
        "results": results
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")

    # 비교 출력
    print("\n" + "=" * 70)
    print("COMPARISON WITH ORIGINAL NIAH")
    print("=" * 70)
    print(f"  Original NIAH (easy):   34.5% (Global 20%, Korean 49%)")
    print(f"  NoLiMa (hard):          {accuracy:.1f}%")
    print(f"  Performance drop:       {34.5 - accuracy:.1f}pp")
    print("=" * 70)

if __name__ == "__main__":
    main()
