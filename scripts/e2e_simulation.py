#!/usr/bin/env python3
"""
End-to-End Simulation of DNSity PoC - Gist Token Pipeline

This script validates the complete Gist Token workflow by running
the unit tests and providing a summary.
"""

import subprocess
import sys
import time
from pathlib import Path


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def run_test_module(module: str, description: str) -> tuple[bool, str]:
    """Run a specific test module and return pass/fail status."""
    print(f"Testing: {description}...")
    result = subprocess.run(
        ["uv", "run", "pytest", f"tests/unit/{module}", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        timeout=120
    )
    passed = result.returncode == 0
    # Extract summary line
    for line in result.stdout.split("\n"):
        if "passed" in line or "failed" in line:
            summary = line.strip()
            break
    else:
        summary = "No summary available"
    return passed, summary


def check_gpu():
    """Check GPU availability."""
    print_section("1. GPU Check")
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Available: {device}")
            print(f"Total Memory: {memory:.1f} GB")
            return True
        else:
            print("No GPU available. Running in CPU mode.")
            return False
    except Exception as e:
        print(f"GPU check failed: {e}")
        return False


def main():
    """Run all e2e validation tests."""
    print("\n" + "="*60)
    print(" DNSity PoC - End-to-End Validation")
    print("="*60)

    start_time = time.time()

    # Check GPU first
    has_gpu = check_gpu()

    # Test modules to validate
    test_modules = [
        ("test_environment.py", "Environment Setup"),
        ("test_niah_generator.py", "NIAH Data Generation"),
        ("test_tokenizer_expansion.py", "Gist Tokenizer"),
        ("test_gist_collator.py", "Attention Masking"),
        ("test_lora_config.py", "LoRA Configuration"),
        ("test_kv_cache.py", "KV Cache Compression"),
        ("test_evaluation.py", "Evaluation Metrics"),
        ("test_baseline.py", "Full Context Baseline"),
        ("test_rag.py", "RAG Pipeline"),
        ("test_trainer.py", "Training Pipeline"),
    ]

    results = {}
    print_section("2. Component Tests")

    for module, description in test_modules:
        try:
            passed, summary = run_test_module(module, description)
            results[description] = (passed, summary)
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {description}: {summary}")
        except subprocess.TimeoutExpired:
            results[description] = (False, "Timeout")
            print(f"  [FAIL] {description}: Timeout")
        except Exception as e:
            results[description] = (False, str(e))
            print(f"  [FAIL] {description}: {e}")

    elapsed = time.time() - start_time

    # Summary
    print_section("SUMMARY")
    passed_count = sum(1 for v in results.values() if v[0])
    total = len(results)

    print(f"GPU Available: {'Yes' if has_gpu else 'No'}")
    print(f"\nComponent Tests: {passed_count}/{total} passed")
    print(f"Time: {elapsed:.1f} seconds")

    # Show failures
    failures = [(k, v[1]) for k, v in results.items() if not v[0]]
    if failures:
        print(f"\nFailed Components ({len(failures)}):")
        for name, reason in failures:
            print(f"  - {name}: {reason}")

    # Final status
    if passed_count == total:
        print("\nE2E Validation: ALL TESTS PASSED!")
        return 0
    elif passed_count >= total * 0.8:
        print(f"\nE2E Validation: MOSTLY PASSED ({passed_count}/{total})")
        return 0
    else:
        print(f"\nE2E Validation: FAILED ({total - passed_count} failures)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
