"""
LoRA configuration for Gist Token training.

CRITICAL: This module ensures that Gist token embeddings are trainable
by including them in modules_to_save.

Without this, the newly added Gist token embeddings would be frozen,
and the model wouldn't learn to compress information into them.
"""

from typing import Optional
from transformers import PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType


def setup_lora_model(
    model: PreTrainedModel,
    num_gist_tokens: int,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
) -> PreTrainedModel:
    """
    Apply LoRA to model with Gist token support.

    Args:
        model: Pre-trained model
        num_gist_tokens: Number of Gist tokens (for documentation)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: LoRA dropout rate
        target_modules: Modules to apply LoRA to (default: q_proj, k_proj, v_proj, o_proj)

    Returns:
        Model with LoRA adapters applied
    """
    if target_modules is None:
        # Default target modules for Llama/GPT models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # GPT-2 uses different names
        # Check if model has c_attn (GPT-2 style)
        if hasattr(model, "transformer") and hasattr(model.transformer.h[0].attn, "c_attn"):
            target_modules = ["c_attn", "c_proj"]

    # CRITICAL: modules_to_save MUST include embedding and lm_head layers
    # This ensures Gist token embeddings are trainable
    # GPT-2 uses "wte" for embeddings, Llama uses "embed_tokens"
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        # GPT-2 style
        modules_to_save = ["wte", "lm_head"]
    else:
        # Llama/Mistral style
        modules_to_save = ["embed_tokens", "lm_head"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=modules_to_save,  # CRITICAL!
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters (for debugging)
    model.print_trainable_parameters()

    return model


def get_trainable_parameter_stats(model: PreTrainedModel) -> dict:
    """
    Get statistics about trainable parameters.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with trainable parameter statistics
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params,
        "trainable_percentage": 100 * trainable_params / total_params,
    }
