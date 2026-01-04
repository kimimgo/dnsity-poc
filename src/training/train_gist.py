"""
Gist Token training utilities.

Provides Trainer setup and configuration for Gist Token fine-tuning.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer
)
from src.model.gist_collator import GistDataCollator


def setup_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: List[Dict[str, Any]],
    max_steps: int = 1000,
    output_dir: str = "checkpoints/gist",
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 100,
    bf16: bool = False,
    num_gist_tokens: Optional[int] = None,
) -> Trainer:
    """
    Setup Hugging Face Trainer for Gist Token fine-tuning.

    Args:
        model: Pre-trained model with Gist tokens added
        tokenizer: Tokenizer with Gist tokens
        train_dataset: List of training samples
        max_steps: Maximum training steps
        output_dir: Directory to save checkpoints
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        bf16: Use bfloat16 precision (if supported)
        num_gist_tokens: Number of Gist tokens (auto-detected if not provided)

    Returns:
        Configured Trainer instance
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-detect num_gist_tokens if not provided
    if num_gist_tokens is None:
        # Count <GIST_N> tokens in tokenizer
        num_gist_tokens = sum(
            1 for token in tokenizer.get_vocab()
            if token.startswith("<GIST_")
        )

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        bf16=bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard for testing
        no_cuda=not torch.cuda.is_available(),  # Use CPU if CUDA unavailable
    )

    # Create data collator
    data_collator = GistDataCollator(
        tokenizer=tokenizer,
        num_gist_tokens=num_gist_tokens
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    return trainer


def get_trainable_parameters(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Get statistics about trainable parameters.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter counts and ratios
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
    }
