"""
Session Keep-Alive Monitor

Prevents session timeout during long-running training by:
1. Monitoring background processes
2. Periodically outputting status updates
3. Detecting training completion
"""

import time
import os
import subprocess
from pathlib import Path
from datetime import datetime


def check_process_running(process_name: str) -> bool:
    """Check if a process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", process_name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False


def get_training_progress() -> dict:
    """Extract training progress from log file."""
    try:
        # Read last line of training output
        result = subprocess.run(
            ["tail", "-1", "/tmp/claude/-home-imgyu-workspace-dnsity-poc/tasks/be993c5.output"],
            capture_output=True,
            text=True
        )

        output = result.stdout

        # Extract step number (e.g., "213/500")
        import re
        match = re.search(r'(\d+)/500', output)
        if match:
            current_step = int(match.group(1))
            progress = (current_step / 500) * 100

            # Extract ETA
            eta_match = re.search(r'\[([0-9:]+)<([0-9:]+),', output)
            elapsed = eta_match.group(1) if eta_match else "unknown"
            remaining = eta_match.group(2) if eta_match else "unknown"

            return {
                "step": current_step,
                "progress": progress,
                "elapsed": elapsed,
                "remaining": remaining,
                "running": True
            }
    except:
        pass

    return {"running": False}


def check_training_complete() -> bool:
    """Check if training has completed."""
    final_checkpoint = Path("checkpoints/gist-global-10/adapter_config.json")
    return final_checkpoint.exists()


def main():
    print("=" * 80)
    print("🔒 Session Keep-Alive Monitor")
    print("=" * 80)
    print("Purpose: Prevent session timeout during long training runs")
    print("Monitoring: Training process + Monitor process")
    print("Update interval: 5 minutes")
    print("=" * 80 + "\n")

    iteration = 0

    while True:
        try:
            iteration += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Check if training is running
            training_running = check_process_running("train_gist_model")
            monitor_running = check_process_running("monitor_training")

            # Get progress
            progress_info = get_training_progress()

            print(f"\n[{timestamp}] Keep-Alive Update #{iteration}")
            print("-" * 80)
            print(f"Training Process: {'✅ Running' if training_running else '❌ Not Running'}")
            print(f"Monitor Process: {'✅ Running' if monitor_running else '❌ Not Running'}")

            if progress_info["running"]:
                print(f"Training Progress: {progress_info['step']}/500 ({progress_info['progress']:.1f}%)")
                print(f"Elapsed: {progress_info['elapsed']} | Remaining: {progress_info['remaining']}")

            # Check if training is complete
            if check_training_complete():
                print("\n" + "=" * 80)
                print("🎉 TRAINING COMPLETED!")
                print("=" * 80)
                print("Monitor will trigger evaluation automatically.")
                print("Keep-alive can now exit.")
                print("=" * 80)
                break

            # If training stopped but not complete, alert
            if not training_running and not check_training_complete():
                print("\n⚠️ WARNING: Training process not running but not complete!")
                print("Check for errors in training log.")

            print("-" * 80)

            # Wait 5 minutes before next update
            time.sleep(300)  # 5 minutes

        except KeyboardInterrupt:
            print("\n\n⚠️ Keep-Alive stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(60)  # Wait 1 minute on error


if __name__ == "__main__":
    main()
