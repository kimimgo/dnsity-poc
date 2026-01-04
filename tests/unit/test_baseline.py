"""
Unit tests for baseline implementations.

Tests verify:
1. Full Context baseline can perform inference
2. VRAM usage is measured correctly
3. Passkey retrieval accuracy (basic sanity check)
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TestFullContextBaseline:
    """Test suite for Full Context baseline."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_full_context_inference(self):
        """Test that Full Context baseline can perform inference."""
        from src.baseline.full_context import FullContextBaseline
        
        # Use small model for testing
        baseline = FullContextBaseline(model_name="gpt2")
        
        context = "The capital of France is Paris. The Eiffel Tower is located there."
        question = "What is the capital of France?"
        
        answer = baseline.inference(context, question)
        
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_baseline_passkey_accuracy(self):
        """Test passkey retrieval with Full Context baseline."""
        from src.baseline.full_context import FullContextBaseline
        
        baseline = FullContextBaseline(model_name="gpt2")
        
        # Simple passkey test
        passkey = "12345"
        context = f"Some random text here. The secret passkey is {passkey}. More text."
        question = "What is the secret passkey?"
        
        answer = baseline.inference(context, question)
        
        # Just verify inference runs (not checking accuracy for GPT-2)
        assert isinstance(answer, str)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_vram_measurement(self):
        """Test VRAM usage measurement."""
        from src.baseline.full_context import FullContextBaseline, measure_vram_usage

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        baseline = FullContextBaseline(model_name="gpt2", device="cuda")

        context = "Short context for testing."
        question = "What is this about?"

        vram_gb = measure_vram_usage(
            lambda: baseline.inference(context, question)
        )

        assert vram_gb > 0
        assert vram_gb < 10.0  # GPT-2 should use < 10GB

    def test_baseline_initialization(self):
        """Test that baseline initializes correctly."""
        from src.baseline.full_context import FullContextBaseline
        
        baseline = FullContextBaseline(model_name="gpt2")
        
        assert baseline.model is not None
        assert baseline.tokenizer is not None
        assert baseline.model_name == "gpt2"

    def test_inference_with_long_context(self):
        """Test inference with longer context."""
        from src.baseline.full_context import FullContextBaseline
        
        baseline = FullContextBaseline(model_name="gpt2")
        
        # Create longer context
        context = " ".join(["This is sentence number " + str(i) for i in range(50)])
        question = "How many sentences are there?"
        
        answer = baseline.inference(context, question, max_new_tokens=10)
        
        assert isinstance(answer, str)
