"""
Unit tests for evaluation metrics and pipelines.

Tests verify:
1. Passkey retrieval accuracy calculation
2. Compression ratio measurement
3. VRAM usage tracking
4. End-to-end evaluation pipeline
"""

import pytest
import torch
from pathlib import Path


class TestPasskeyEvaluation:
    """Test suite for Passkey Retrieval (NIAH) evaluation."""

    def test_extract_passkey_from_text(self):
        """Test passkey extraction from generated text."""
        from src.evaluation.metrics import extract_passkey
        
        generated = "The answer is 12345. This is the passkey."
        passkey = extract_passkey(generated, pattern=r'\d{5}')
        
        assert passkey == "12345"

    def test_passkey_accuracy_calculation(self):
        """Test accuracy calculation for passkey retrieval."""
        from src.evaluation.metrics import calculate_passkey_accuracy
        
        results = [
            {"predicted": "12345", "ground_truth": "12345"},
            {"predicted": "67890", "ground_truth": "67890"},
            {"predicted": "11111", "ground_truth": "22222"},  # Wrong
        ]
        
        accuracy = calculate_passkey_accuracy(results)
        
        assert accuracy == pytest.approx(2/3, rel=0.01)

    def test_niah_evaluator_initialization(self):
        """Test NIAH evaluator initialization."""
        from src.evaluation.niah_evaluator import NIAHEvaluator
        
        evaluator = NIAHEvaluator(model_name="gpt2")
        
        assert evaluator.model is not None
        assert evaluator.tokenizer is not None

    def test_evaluate_single_sample(self):
        """Test evaluating a single NIAH sample."""
        from src.evaluation.niah_evaluator import NIAHEvaluator
        
        evaluator = NIAHEvaluator(model_name="gpt2")
        
        sample = {
            "context": "Some text. The passkey is 12345. More text.",
            "question": "What is the passkey?",
            "answer": "12345"
        }
        
        result = evaluator.evaluate_sample(sample, max_new_tokens=20)
        
        assert "predicted" in result
        assert "ground_truth" in result
        assert result["ground_truth"] == "12345"


class TestCompressionMetrics:
    """Test suite for compression and VRAM metrics."""

    def test_calculate_compression_ratio(self):
        """Test compression ratio calculation."""
        from src.evaluation.metrics import calculate_compression_ratio
        
        original_length = 4000
        compressed_length = 10
        
        ratio = calculate_compression_ratio(original_length, compressed_length)
        
        assert ratio == 400.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_measure_vram_usage(self):
        """Test VRAM usage measurement."""
        from src.evaluation.metrics import measure_vram_mb
        
        def dummy_operation():
            x = torch.randn(1000, 1000, device="cuda")
            y = x @ x.T
            return y.sum().item()
        
        vram_mb = measure_vram_mb(dummy_operation)
        
        assert vram_mb > 0
        assert vram_mb < 1000  # Should be < 1GB for this small operation

    def test_calculate_throughput(self):
        """Test tokens/sec calculation."""
        from src.evaluation.metrics import calculate_throughput
        
        num_tokens = 100
        elapsed_time = 2.5  # seconds
        
        throughput = calculate_throughput(num_tokens, elapsed_time)
        
        assert throughput == 40.0  # 100/2.5
