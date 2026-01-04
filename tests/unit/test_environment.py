"""
Environment validation tests.

Ensures that the development environment is properly configured with
GPU support, correct library versions, and necessary dependencies.
"""

import pytest
import sys


class TestEnvironment:
    """Test suite for environment validation."""

    def test_python_version(self):
        """Test that Python version is 3.10 or higher."""
        assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version}"

    def test_torch_import(self):
        """Test that PyTorch is installed."""
        import torch
        assert torch.__version__ is not None

    def test_transformers_import(self):
        """Test that transformers library is installed."""
        import transformers
        # Check version is 4.36.0 or higher
        version_parts = transformers.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        assert (major, minor) >= (4, 36), f"transformers >= 4.36.0 required, got {transformers.__version__}"

    def test_cuda_available(self):
        """Test that CUDA is available for GPU training."""
        import torch
        assert torch.cuda.is_available(), "CUDA not available - GPU required for this project"

    def test_gpu_memory(self):
        """Test that GPU has sufficient memory (ideally 24GB)."""
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            # Get GPU memory in GB
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9

            # Warn if less than 20GB (some overhead expected)
            assert total_memory >= 20, f"GPU has {total_memory:.1f}GB, 24GB recommended for this project"

    def test_bitsandbytes_available(self):
        """Test that bitsandbytes is installed for 4-bit quantization."""
        import bitsandbytes
        assert bitsandbytes.__version__ is not None

    def test_peft_available(self):
        """Test that PEFT library is installed for LoRA."""
        import peft
        assert peft.__version__ is not None

    def test_datasets_available(self):
        """Test that datasets library is installed."""
        import datasets
        assert datasets.__version__ is not None

    def test_huggingface_hub_available(self):
        """Test that huggingface_hub is installed."""
        import huggingface_hub
        assert huggingface_hub.__version__ is not None

    def test_model_loading_capability(self):
        """Test ability to load a small model (smoke test)."""
        import torch
        from transformers import AutoTokenizer

        # Try loading a tiny tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        assert tokenizer is not None
        assert len(tokenizer) > 0

    def test_quantization_config(self):
        """Test that 4-bit quantization config can be created."""
        from transformers import BitsAndBytesConfig
        import torch

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"

    def test_gpu_tensor_operations(self):
        """Test basic GPU tensor operations."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")

        # Create tensors on GPU
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)

        # Perform operation
        z = torch.matmul(x, y)

        assert z.device.type == "cuda"
        assert z.shape == (1000, 1000)


class TestProjectStructure:
    """Test that project structure is correctly set up."""

    def test_src_directory_exists(self):
        """Test that src directory exists."""
        from pathlib import Path
        assert (Path("src")).exists()

    def test_tests_directory_exists(self):
        """Test that tests directory exists."""
        from pathlib import Path
        assert (Path("tests")).exists()

    def test_data_directory_exists(self):
        """Test that data directory exists."""
        from pathlib import Path
        assert (Path("data")).exists()

    def test_src_modules_importable(self):
        """Test that src modules can be imported."""
        from src.data import download_longbench
        from src.data import create_niah

        assert download_longbench is not None
        assert create_niah is not None


