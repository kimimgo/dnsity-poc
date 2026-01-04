#!/usr/bin/env python3
"""
Environment validation script.

Checks that all required dependencies are installed and GPU is available.
Run this before starting training.
"""

import sys
import subprocess


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version >= (3, 10):
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (need 3.10+)")
        return False


def check_package(package_name, min_version=None):
    """Check if a package is installed."""
    try:
        module = __import__(package_name)
        version = getattr(module, "__version__", "unknown")

        if min_version and version != "unknown":
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"  ⚠️  {package_name} {version} (need {min_version}+)")
                return False

        print(f"  ✅ {package_name} {version}")
        return True
    except ImportError:
        print(f"  ❌ {package_name} not installed")
        return False


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            props = torch.cuda.get_device_properties(device)
            memory_gb = props.total_memory / 1e9
            print(f"  ✅ CUDA available")
            print(f"     GPU: {props.name}")
            print(f"     Memory: {memory_gb:.1f} GB")

            if memory_gb < 20:
                print(f"     ⚠️  Less than 24GB - may have memory issues")
            return True
        else:
            print(f"  ❌ CUDA not available")
            return False
    except Exception as e:
        print(f"  ❌ Error checking CUDA: {e}")
        return False


def check_huggingface_auth():
    """Check Hugging Face authentication."""
    print("\nChecking Hugging Face authentication...")
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            username = result.stdout.strip()
            print(f"  ✅ Logged in as: {username}")
            return True
        else:
            print(f"  ⚠️  Not logged in (optional, but recommended)")
            return True  # Not critical
    except FileNotFoundError:
        print(f"  ⚠️  huggingface-cli not found")
        return True  # Not critical


def main():
    """Run all environment checks."""
    print("=" * 60)
    print("DNSity PoC - Environment Validation")
    print("=" * 60)

    checks = []

    # Python version
    checks.append(check_python_version())

    # Core packages
    print("\nChecking core packages...")
    checks.append(check_package("torch"))
    checks.append(check_package("transformers", "4.36.0"))
    checks.append(check_package("datasets"))
    checks.append(check_package("huggingface_hub"))

    # Training packages (may not be installed yet)
    print("\nChecking training packages (optional for Phase 1)...")
    check_package("bitsandbytes")
    check_package("peft")
    check_package("accelerate")

    # Utility packages
    print("\nChecking utility packages...")
    checks.append(check_package("numpy"))
    checks.append(check_package("pandas"))

    # CUDA
    checks.append(check_cuda())

    # Hugging Face
    check_huggingface_auth()

    # Summary
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ All critical checks passed!")
        print("\nYou can proceed with:")
        print("  python src/data/download_longbench.py")
        print("  python src/data/create_niah.py")
        return 0
    else:
        print("❌ Some checks failed")
        print("\nInstall missing packages:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
