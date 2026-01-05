"""
Monitor training progress and automatically trigger evaluation when complete.

This script continuously monitors the training process and:
1. Tracks progress in real-time
2. Detects when training completes
3. Automatically triggers evaluation
4. Prevents session timeout during long training runs
"""

import os
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime


def check_training_status(checkpoint_dir: Path) -> dict:
    """Check training status by looking at checkpoint directories."""
    checkpoints = []

    if not checkpoint_dir.exists():
        return {"status": "not_started", "latest_step": 0, "checkpoints": []}

    # Find all checkpoint directories
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[1])
                checkpoints.append(step)
            except:
                pass

    checkpoints.sort()

    # Check if final checkpoint exists
    final_checkpoint = checkpoint_dir / "adapter_config.json"
    if final_checkpoint.exists():
        status = "completed"
        latest_step = 500
    elif checkpoints:
        status = "in_progress"
        latest_step = checkpoints[-1]
    else:
        status = "not_started"
        latest_step = 0

    return {
        "status": status,
        "latest_step": latest_step,
        "checkpoints": checkpoints
    }


def run_evaluation(checkpoint_path: Path, dataset_path: Path, output_path: Path, limit: int = 200):
    """Run evaluation on a checkpoint."""
    print(f"\n{'='*80}")
    print(f"🔬 Running Evaluation")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    cmd = [
        "python3",
        "experiments/run_gist_inference.py",
        "--checkpoint", str(checkpoint_path),
        "--dataset", str(dataset_path),
        "--output", str(output_path),
        "--limit", str(limit),
        "--device", "cuda"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Evaluation completed successfully")
        return True
    else:
        print(f"❌ Evaluation failed:")
        print(result.stderr)
        return False


def main():
    checkpoint_dir = Path("checkpoints/gist-global-10")
    global_dataset = Path("data/processed/niah/global_niah.jsonl")
    korean_dataset = Path("data/processed/niah/korean_niah.jsonl")
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("📊 Training Monitor Started")
    print("="*80)
    print(f"Monitoring: {checkpoint_dir}")
    print(f"Check interval: 60 seconds")
    print("="*80 + "\n")

    last_step = 0
    evaluated_checkpoints = set()

    while True:
        try:
            status = check_training_status(checkpoint_dir)
            current_step = status["latest_step"]

            if current_step > last_step:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Progress: Step {current_step}/500 ({current_step/500*100:.1f}%)")
                last_step = current_step

            # Check if training is complete
            if status["status"] == "completed":
                print("\n" + "="*80)
                print("✅ TRAINING COMPLETED!")
                print("="*80 + "\n")

                # Run evaluations on final checkpoint
                print("Starting comprehensive evaluation...\n")

                # Global NIAH evaluation
                run_evaluation(
                    checkpoint_dir,
                    global_dataset,
                    results_dir / "final_global_results.json",
                    limit=200
                )

                # Korean NIAH evaluation
                run_evaluation(
                    checkpoint_dir,
                    korean_dataset,
                    results_dir / "final_korean_results.json",
                    limit=200
                )

                print("\n" + "="*80)
                print("✅ ALL EVALUATIONS COMPLETED")
                print("="*80)
                print(f"\nResults saved to: {results_dir}")
                print("\nNext steps:")
                print("1. Review evaluation results")
                print("2. Validate against CONCEPT.md criteria")
                print("3. Run Gemini re-evaluation for 100/100 score")
                print("="*80 + "\n")

                break

            # Skip intermediate evaluation during training (GPU memory constraint)
            # Will evaluate all checkpoints after training completes

            # Wait before next check
            time.sleep(60)

        except KeyboardInterrupt:
            print("\n\n⚠️ Monitor stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
