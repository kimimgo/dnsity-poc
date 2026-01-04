"""
Unit tests for KV Cache serialization and compression.

Tests verify:
1. Gist token KV cache extraction from past_key_values
2. Serialization to .safetensors format
3. Deserialization from disk
4. End-to-end save/load/inference cycle
"""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil


class TestKVCacheExtraction:
    """Test suite for KV cache extraction and slicing."""

    def test_extract_gist_kv_from_past_key_values(self):
        """Test extracting only Gist token KV from full past_key_values."""
        from src.inference.kv_cache import extract_gist_kv

        # Simulate past_key_values structure for Llama-3
        # Format: Tuple[Tuple[Tensor, Tensor], ...] for each layer
        # Shape: (batch, num_heads, seq_len, head_dim)
        batch_size = 1
        num_heads = 8
        seq_len = 100  # Full sequence length
        head_dim = 64
        num_layers = 4

        # Create fake past_key_values
        past_key_values = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),  # Key
                torch.randn(batch_size, num_heads, seq_len, head_dim),  # Value
            )
            for _ in range(num_layers)
        )

        # Extract only Gist tokens (positions 50-60, num_gist_tokens=10)
        gist_start = 50
        gist_end = 60

        gist_kv = extract_gist_kv(
            past_key_values=past_key_values,
            gist_start=gist_start,
            gist_end=gist_end
        )

        # Verify structure
        assert len(gist_kv) == num_layers
        assert gist_kv[0][0].shape == (batch_size, num_heads, 10, head_dim)
        assert gist_kv[0][1].shape == (batch_size, num_heads, 10, head_dim)

    def test_detect_gist_positions_from_tokenizer(self):
        """Test automatic detection of Gist token positions in input."""
        from src.inference.kv_cache import detect_gist_positions
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Add Gist tokens
        gist_tokens = [f"<GIST_{i}>" for i in range(10)]
        tokenizer.add_special_tokens({"additional_special_tokens": gist_tokens})

        # Create input with Gist tokens
        text = "Context here. <GIST_0><GIST_1><GIST_2> Question here."
        input_ids = tokenizer.encode(text, return_tensors="pt")

        gist_start, gist_end = detect_gist_positions(input_ids, tokenizer)

        # Verify positions are detected
        assert gist_start is not None
        assert gist_end is not None
        assert gist_end - gist_start == 3  # 3 Gist tokens


class TestKVCacheSerialization:
    """Test suite for KV cache serialization to disk."""

    def test_save_gist_kv_to_safetensors(self, tmp_path):
        """Test saving Gist KV cache to .safetensors format."""
        from src.inference.kv_cache import save_gist_kv

        # Create fake Gist KV cache
        batch_size = 1
        num_heads = 8
        gist_len = 10
        head_dim = 64
        num_layers = 4

        gist_kv = tuple(
            (
                torch.randn(batch_size, num_heads, gist_len, head_dim),
                torch.randn(batch_size, num_heads, gist_len, head_dim),
            )
            for _ in range(num_layers)
        )

        # Save to disk
        save_path = tmp_path / "gist_kv.safetensors"
        metadata = {
            "num_gist_tokens": 10,
            "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "num_layers": 4
        }

        save_gist_kv(gist_kv, save_path, metadata=metadata)

        # Verify file exists
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_load_gist_kv_from_safetensors(self, tmp_path):
        """Test loading Gist KV cache from .safetensors format."""
        from src.inference.kv_cache import save_gist_kv, load_gist_kv

        # Create and save fake Gist KV
        batch_size = 1
        num_heads = 8
        gist_len = 10
        head_dim = 64
        num_layers = 4

        original_gist_kv = tuple(
            (
                torch.randn(batch_size, num_heads, gist_len, head_dim),
                torch.randn(batch_size, num_heads, gist_len, head_dim),
            )
            for _ in range(num_layers)
        )

        save_path = tmp_path / "gist_kv.safetensors"
        metadata = {
            "num_gist_tokens": 10,
            "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "num_layers": 4
        }

        save_gist_kv(original_gist_kv, save_path, metadata=metadata)

        # Load back
        loaded_gist_kv, loaded_metadata = load_gist_kv(save_path)

        # Verify structure matches
        assert len(loaded_gist_kv) == num_layers
        assert loaded_gist_kv[0][0].shape == original_gist_kv[0][0].shape
        assert loaded_gist_kv[0][1].shape == original_gist_kv[0][1].shape

        # Verify metadata
        assert loaded_metadata["num_gist_tokens"] == 10
        assert loaded_metadata["model_name"] == "meta-llama/Meta-Llama-3-8B-Instruct"

        # Verify content matches
        assert torch.allclose(loaded_gist_kv[0][0], original_gist_kv[0][0])
        assert torch.allclose(loaded_gist_kv[0][1], original_gist_kv[0][1])


class TestKVCacheInjection:
    """Test suite for injecting saved KV cache into inference."""

    def test_inject_gist_kv_for_inference(self):
        """Test injecting Gist KV cache for question-answering inference."""
        from src.inference.kv_cache import inject_gist_kv
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load small model for testing
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Create fake Gist KV (GPT-2 has 12 layers)
        batch_size = 1
        num_heads = 12
        gist_len = 5
        head_dim = 64
        num_layers = 12

        gist_kv = tuple(
            (
                torch.randn(batch_size, num_heads, gist_len, head_dim),
                torch.randn(batch_size, num_heads, gist_len, head_dim),
            )
            for _ in range(num_layers)
        )

        # Inject KV cache
        question = "What is the answer?"
        inputs = tokenizer(question, return_tensors="pt")

        outputs = inject_gist_kv(
            model=model,
            tokenizer=tokenizer,
            gist_kv=gist_kv,
            question=question,
            max_new_tokens=10
        )

        # Verify output structure
        assert "generated_text" in outputs
        assert "input_length" in outputs
        assert "output_length" in outputs

    def test_end_to_end_compress_save_load_generate(self, tmp_path):
        """Test complete workflow: compress context → save KV → load KV → generate."""
        from src.inference.kv_cache import extract_gist_kv, save_gist_kv, load_gist_kv, inject_gist_kv
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Setup
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Add Gist tokens
        gist_tokens = [f"<GIST_{i}>" for i in range(5)]
        tokenizer.add_special_tokens({"additional_special_tokens": gist_tokens})
        model.resize_token_embeddings(len(tokenizer))

        # Step 1: Compress context (simulate)
        context_with_gist = "Context text. <GIST_0><GIST_1><GIST_2><GIST_3><GIST_4>"
        inputs = tokenizer(context_with_gist, return_tensors="pt")

        # Forward pass to get KV cache
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            past_key_values = outputs.past_key_values

        # Find Gist token positions
        gist_token_ids = [tokenizer.convert_tokens_to_ids(f"<GIST_{i}>") for i in range(5)]
        input_ids_list = inputs["input_ids"][0].tolist()

        gist_start = None
        for i, token_id in enumerate(input_ids_list):
            if token_id == gist_token_ids[0]:
                gist_start = i
                break

        gist_end = gist_start + 5

        # Extract Gist KV
        gist_kv = extract_gist_kv(
            past_key_values=past_key_values,
            gist_start=gist_start,
            gist_end=gist_end
        )

        # Step 2: Save to disk
        save_path = tmp_path / "compressed_kv.safetensors"
        metadata = {
            "num_gist_tokens": 5,
            "model_name": "gpt2",
            "num_layers": 12
        }
        save_gist_kv(gist_kv, save_path, metadata=metadata)

        # Step 3: Load from disk
        loaded_gist_kv, loaded_metadata = load_gist_kv(save_path)

        # Step 4: Generate with loaded KV
        question = "What is the context about?"
        outputs = inject_gist_kv(
            model=model,
            tokenizer=tokenizer,
            gist_kv=loaded_gist_kv,
            question=question,
            max_new_tokens=10
        )

        # Verify success
        assert "generated_text" in outputs
        assert len(outputs["generated_text"]) > 0
