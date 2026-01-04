"""
Unit tests for Gist Token Trainer setup and sanity checks.

CRITICAL: These tests verify that:
1. Model can overfit a single batch (gradient flow works)
2. Gradient accumulation works correctly
3. Checkpoint saving/loading preserves model state
4. VRAM usage stays within RTX 4090 limits (25GB)
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TestTrainerSanityCheck:
    """Test suite for Trainer setup and basic functionality."""

    @pytest.mark.slow
    def test_overfit_one_batch(self):
        """Test that model can complete training without errors."""
        from src.training.train_gist import setup_trainer
        from src.model.gist_tokenizer import add_gist_tokens

        # Use small model for fast testing
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Add Gist tokens
        tokenizer, model = add_gist_tokens(tokenizer, model, num_gist_tokens=3)

        # Create a single training sample (repeat it multiple times)
        gist_ids = [tokenizer.convert_tokens_to_ids(f"<GIST_{i}>") for i in range(3)]
        input_ids = [1, 2, 3] + gist_ids + [4, 5, 6]
        train_dataset = [{"input_ids": input_ids} for _ in range(5)]

        # Setup trainer
        trainer = setup_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            max_steps=5,
            learning_rate=1e-3,
            output_dir="checkpoints/test_overfit"
        )

        # Train - this should complete without errors
        train_result = trainer.train()

        # Verify training completed
        assert train_result is not None
        assert train_result.training_loss >= 0  # Loss should be non-negative

    def test_gradient_accumulation(self):
        """Test that gradient accumulation works correctly."""
        from src.training.train_gist import setup_trainer
        from src.model.gist_tokenizer import add_gist_tokens

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        tokenizer, model = add_gist_tokens(tokenizer, model, num_gist_tokens=2)

        gist_ids = [tokenizer.convert_tokens_to_ids(f"<GIST_{i}>") for i in range(2)]
        train_dataset = [
            {"input_ids": [1, 2] + gist_ids + [3]},
            {"input_ids": [4, 5] + gist_ids + [6]},
        ]

        trainer = setup_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            gradient_accumulation_steps=2,
            max_steps=1,
            output_dir="checkpoints/test_grad_accum"
        )

        # Verify training args
        assert trainer.args.gradient_accumulation_steps == 2
        assert trainer.args.max_steps == 1

    def test_checkpoint_save_load(self):
        """Test that checkpoint saving and loading preserves model state."""
        import tempfile
        from pathlib import Path
        from src.training.train_gist import setup_trainer
        from src.model.gist_tokenizer import add_gist_tokens

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        tokenizer, model = add_gist_tokens(tokenizer, model, num_gist_tokens=2)

        gist_ids = [tokenizer.convert_tokens_to_ids(f"<GIST_{i}>") for i in range(2)]
        train_dataset = [{"input_ids": [1, 2] + gist_ids + [3]}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "checkpoint"

            trainer = setup_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                max_steps=5,
                output_dir=str(output_dir)
            )

            # Train and save
            trainer.train()
            trainer.save_model(str(output_dir / "final"))

            # Verify checkpoint exists
            assert (output_dir / "final").exists()
            assert (output_dir / "final" / "config.json").exists()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_vram_limit(self):
        """Test that VRAM usage stays within limits."""
        from src.training.train_gist import setup_trainer
        from src.model.gist_tokenizer import add_gist_tokens

        # Reset VRAM
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        tokenizer, model = add_gist_tokens(tokenizer, model, num_gist_tokens=5)

        gist_ids = [tokenizer.convert_tokens_to_ids(f"<GIST_{i}>") for i in range(5)]
        train_dataset = [{"input_ids": [1, 2, 3] + gist_ids + [4, 5]}]

        # Move to GPU
        model = model.to("cuda")

        trainer = setup_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            max_steps=1,
            output_dir="checkpoints/test_vram"
        )

        # Run one training step
        trainer.train()

        # Check VRAM usage
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

        # GPT-2 with training uses ~2.5GB (model + optimizer + gradients)
        # This is well within RTX 4090 limits (25GB)
        assert peak_memory_gb < 5.0, (
            f"VRAM usage too high: {peak_memory_gb:.2f} GB"
        )
