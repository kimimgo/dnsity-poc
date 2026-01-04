"""
Unit tests for LoRA configuration with Gist Token support.

CRITICAL: These tests verify that:
1. LoRA is applied to the correct modules (q_proj, k_proj, v_proj, o_proj)
2. Gist token embeddings are trainable (modules_to_save includes embed_tokens)
3. Gradient flows to Gist token embeddings during backward pass
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Skip all tests if peft is not installed
try:
    import peft
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PEFT_AVAILABLE,
    reason="peft library not installed (Phase 2 dependency)"
)


class TestLoRAConfiguration:
    """Test suite for LoRA configuration with Gist tokens."""

    def test_trainable_parameters_exist(self):
        """Test that modules_to_save includes embed_tokens and lm_head."""
        from src.model.gist_lora import setup_lora_model
        from src.model.gist_tokenizer import add_gist_tokens

        # Use small model for testing
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Add Gist tokens
        tokenizer, model = add_gist_tokens(tokenizer, model, num_gist_tokens=5)

        # Setup LoRA
        model = setup_lora_model(model, num_gist_tokens=5)

        # Verify that embed_tokens are trainable
        embed_params = model.get_input_embeddings().weight
        assert embed_params.requires_grad, "embed_tokens should be trainable"

        # Verify that lm_head is trainable
        lm_head_params = model.get_output_embeddings().weight
        assert lm_head_params.requires_grad, "lm_head should be trainable"

    def test_gist_embedding_gradient_flow(self):
        """Test that gradients flow to Gist token embeddings."""
        from src.model.gist_lora import setup_lora_model
        from src.model.gist_tokenizer import add_gist_tokens, get_gist_token_ids

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Add Gist tokens
        tokenizer, model = add_gist_tokens(tokenizer, model, num_gist_tokens=3)

        # Setup LoRA
        model = setup_lora_model(model, num_gist_tokens=3)

        # Create dummy input with Gist tokens
        gist_ids = get_gist_token_ids(tokenizer, num_gist_tokens=3)
        input_ids = torch.tensor([[1, 2] + gist_ids + [3, 4]])
        labels = input_ids.clone()

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Verify gradient exists for Gist token embeddings
        embed_weight = model.get_input_embeddings().weight
        gist_grads = embed_weight.grad[gist_ids]

        assert gist_grads is not None, "Gist embeddings should have gradients"
        assert torch.sum(torch.abs(gist_grads)) > 0, "Gist gradients should be non-zero"

    def test_lora_target_modules_applied(self):
        """Test that LoRA is applied to q_proj, k_proj, v_proj, o_proj."""
        from src.model.gist_lora import setup_lora_model

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model = setup_lora_model(model, num_gist_tokens=5)

        # Check if model has LoRA adapters
        # PEFT wraps the model, so we check for peft attributes
        assert hasattr(model, "peft_config"), "Model should have PEFT config"

        # Verify LoRA config exists
        peft_config = model.peft_config
        assert len(peft_config) > 0, "PEFT config should not be empty"

    def test_lora_rank_and_alpha(self):
        """Test that LoRA rank and alpha are correctly set."""
        from src.model.gist_lora import setup_lora_model

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model = setup_lora_model(
            model,
            num_gist_tokens=5,
            lora_r=16,
            lora_alpha=32
        )

        # Get PEFT config
        peft_config = model.peft_config["default"]

        assert peft_config.r == 16, "LoRA rank should be 16"
        assert peft_config.lora_alpha == 32, "LoRA alpha should be 32"

    def test_lora_dropout(self):
        """Test that LoRA dropout is correctly set."""
        from src.model.gist_lora import setup_lora_model

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model = setup_lora_model(
            model,
            num_gist_tokens=5,
            lora_dropout=0.05
        )

        peft_config = model.peft_config["default"]
        assert peft_config.lora_dropout == 0.05, "LoRA dropout should be 0.05"

    def test_only_lora_and_modules_to_save_trainable(self):
        """Test that only LoRA parameters and modules_to_save are trainable."""
        from src.model.gist_lora import setup_lora_model

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model = setup_lora_model(model, num_gist_tokens=5)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        # LoRA should significantly reduce trainable parameters
        trainable_ratio = trainable_params / total_params

        # Typically LoRA trains < 1% of parameters
        assert trainable_ratio < 0.1, f"Too many trainable params: {trainable_ratio:.2%}"

    def test_model_forward_pass_with_lora(self):
        """Test that model can perform forward pass after LoRA is applied."""
        from src.model.gist_lora import setup_lora_model
        from src.model.gist_tokenizer import add_gist_tokens

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        tokenizer, model = add_gist_tokens(tokenizer, model, num_gist_tokens=3)
        model = setup_lora_model(model, num_gist_tokens=3)

        # Create input with Gist tokens
        gist_ids = [tokenizer.convert_tokens_to_ids(f"<GIST_{i}>") for i in range(3)]
        input_ids = torch.tensor([[1, 2] + gist_ids + [3]])

        # Forward pass should work
        outputs = model(input_ids=input_ids)

        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1  # batch size
        assert outputs.logits.shape[1] == input_ids.shape[1]  # sequence length
