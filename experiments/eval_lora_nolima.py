"""
LoRA Context Injection NoLiMa 평가

두 가지 모드:
1. Direct Inference: Context를 직접 입력 (Full Context baseline)
2. LoRA Fine-tuning: 데이터셋으로 LoRA 학습 후 평가

NoLiMa에서 LoRA가 어휘 분리된 질문에 얼마나 잘 대응하는지 테스트
"""
import json
import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import argparse
from tqdm import tqdm
import gc


def load_base_model(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """베이스 모델 로드"""
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True
    )

    return model, tokenizer


def evaluate_direct_inference(model, tokenizer, samples, max_ctx_len=3500):
    """Direct Inference: Context를 직접 입력하여 평가 (Full Context baseline)"""
    print("\n=== Direct Inference (Full Context) Mode ===")

    results = []
    for sample in tqdm(samples, desc="Direct Inference"):
        try:
            # 컨텍스트 자르기
            ctx_tokens = tokenizer.encode(sample["context"], add_special_tokens=False)
            if len(ctx_tokens) > max_ctx_len:
                ctx_tokens = ctx_tokens[:max_ctx_len]
            context = tokenizer.decode(ctx_tokens)

            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {sample['question']}

Answer (provide only the code/identifier mentioned):"""

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

            results.append({
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": correct,
                "generated": generated[:100],
                "question": sample["question"]
            })

        except Exception as e:
            print(f"\nError: {e}")
            results.append({
                "predicted": "",
                "ground_truth": sample["answer"],
                "correct": False,
                "error": str(e)
            })

    return results


def prepare_lora_data(samples, tokenizer, max_ctx_len=2000):
    """LoRA 학습용 데이터 준비"""
    formatted_data = []

    for sample in samples:
        ctx_tokens = tokenizer.encode(sample["context"], add_special_tokens=False)
        if len(ctx_tokens) > max_ctx_len:
            ctx_tokens = ctx_tokens[:max_ctx_len]
        context = tokenizer.decode(ctx_tokens)

        # 학습 텍스트 (context + question + answer)
        text = f"""Context: {context}

Question: {sample['question']}

Answer: {sample['answer']}"""

        formatted_data.append({"text": text, "answer": sample["answer"], "question": sample["question"]})

    return formatted_data


def train_lora(model, tokenizer, train_data, output_dir="checkpoints/lora-nolima-temp"):
    """LoRA 학습"""
    print("\n=== LoRA Fine-tuning ===")

    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 데이터셋 준비
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length"
        )
        # labels 추가 (input_ids와 동일, 패딩은 -100으로)
        tokenized["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in ids]
            for ids in tokenized["input_ids"]
        ]
        return tokenized

    dataset = Dataset.from_list(train_data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text", "answer", "question"])

    # 학습 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="no",
        fp16=False,
        bf16=True,
        report_to="none",
    )

    # 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    return model


def evaluate_lora_model(model, tokenizer, samples, max_ctx_len=3500):
    """LoRA 학습된 모델로 평가 - 컨텍스트 포함"""
    print("\n=== LoRA Model Evaluation (with context) ===")

    results = []
    for sample in tqdm(samples, desc="LoRA Evaluation"):
        try:
            # 컨텍스트 + 질문 입력 (LoRA가 추출 능력을 향상시켰는지 확인)
            ctx_tokens = tokenizer.encode(sample["context"], add_special_tokens=False)
            if len(ctx_tokens) > max_ctx_len:
                ctx_tokens = ctx_tokens[:max_ctx_len]
            context = tokenizer.decode(ctx_tokens)

            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {sample['question']}

Answer (provide only the code/identifier mentioned):"""

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

            results.append({
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": correct,
                "generated": generated[:100],
                "question": sample["question"]
            })

        except Exception as e:
            print(f"\nError: {e}")
            results.append({
                "predicted": "",
                "ground_truth": sample["answer"],
                "correct": False,
                "error": str(e)
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/nolima/nolima_200.jsonl")
    parser.add_argument("--output", default="results/lora_nolima.json")
    parser.add_argument("--mode", choices=["direct", "lora", "both"], default="both",
                       help="direct: Full context inference, lora: LoRA fine-tuning, both: Run both")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=50,
                       help="Number of samples for LoRA training (rest for eval)")
    args = parser.parse_args()

    # 모델 로드
    model, tokenizer = load_base_model()

    # 데이터 로드
    print(f"Loading {args.dataset}")
    with open(args.dataset) as f:
        samples = [json.loads(line) for line in f]

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Total samples: {len(samples)}")

    output_data = {
        "dataset": args.dataset,
        "mode": args.mode,
        "total_samples": len(samples),
    }

    # Direct Inference (Full Context baseline)
    if args.mode in ["direct", "both"]:
        direct_results = evaluate_direct_inference(model, tokenizer, samples)
        direct_correct = sum(r["correct"] for r in direct_results)
        direct_accuracy = direct_correct / len(direct_results) * 100

        print("\n" + "=" * 70)
        print("DIRECT INFERENCE (Full Context) - NoLiMa RESULTS")
        print("=" * 70)
        print(f"Accuracy: {direct_accuracy:.2f}% ({direct_correct}/{len(direct_results)})")
        print("=" * 70)

        output_data["direct_inference"] = {
            "accuracy": direct_accuracy,
            "correct": direct_correct,
            "total": len(direct_results),
            "results": direct_results[:20]  # 처음 20개만 저장
        }

    # LoRA Fine-tuning
    if args.mode in ["lora", "both"]:
        # 학습/테스트 분할
        train_samples = samples[:args.train_samples]
        test_samples = samples[args.train_samples:] if len(samples) > args.train_samples else samples

        print(f"\nLoRA Training: {len(train_samples)} samples")
        print(f"LoRA Testing: {len(test_samples)} samples")

        # 학습 데이터 준비
        train_data = prepare_lora_data(train_samples, tokenizer)

        # GPU 메모리 정리
        gc.collect()
        torch.cuda.empty_cache()

        # LoRA 학습
        model = train_lora(model, tokenizer, train_data)

        # 평가 (학습에 사용한 샘플 + 사용하지 않은 샘플 둘 다)
        print("\n--- Testing on TRAINING samples (memorization check) ---")
        train_results = evaluate_lora_model(model, tokenizer, train_samples)
        train_correct = sum(r["correct"] for r in train_results)
        train_accuracy = train_correct / len(train_results) * 100
        print(f"Training set accuracy: {train_accuracy:.2f}% ({train_correct}/{len(train_results)})")

        print("\n--- Testing on UNSEEN samples (generalization check) ---")
        lora_results = evaluate_lora_model(model, tokenizer, test_samples)
        lora_correct = sum(r["correct"] for r in lora_results)
        lora_accuracy = lora_correct / len(lora_results) * 100

        print("\n" + "=" * 70)
        print("LoRA FINE-TUNING - NoLiMa RESULTS")
        print("=" * 70)
        print(f"Training samples: {len(train_samples)}")
        print(f"Test samples: {len(test_samples)}")
        print(f"Accuracy: {lora_accuracy:.2f}% ({lora_correct}/{len(lora_results)})")
        print("=" * 70)

        # 오답 분석
        print("\nIncorrect samples (first 5):")
        incorrect = [r for r in lora_results if not r["correct"]][:5]
        for i, r in enumerate(incorrect):
            print(f"  [{i+1}] Q: {r.get('question', '')[:60]}...")
            print(f"      Expected: {r['ground_truth']}, Got: {r['predicted']}")

        output_data["lora_finetuning"] = {
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "train_accuracy": train_accuracy,
            "train_correct": train_correct,
            "test_accuracy": lora_accuracy,
            "test_correct": lora_correct,
            "total": len(lora_results),
            "results": lora_results[:20]
        }

    # 결과 저장
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")

    # 비교 출력
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    if "direct_inference" in output_data:
        print(f"  Direct Inference (Full Context): {output_data['direct_inference']['accuracy']:.1f}%")
    if "lora_finetuning" in output_data:
        print(f"  LoRA Train Accuracy:              {output_data['lora_finetuning']['train_accuracy']:.1f}%")
        print(f"  LoRA Test Accuracy:               {output_data['lora_finetuning']['test_accuracy']:.1f}%")
    print("  RAG (reference):                  94.5%")
    print("  Gist Token (reference):           41.5%")
    print("=" * 70)


if __name__ == "__main__":
    main()
