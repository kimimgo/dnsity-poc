"""
Gist Token tokenizer utilities.

Provides functions to add special Gist tokens to tokenizers and resize model embeddings.
"""

from typing import Tuple, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel


def add_gist_tokens(
    tokenizer: PreTrainedTokenizer,
    model: Optional[PreTrainedModel],
    num_gist_tokens: int
) -> Tuple[PreTrainedTokenizer, Optional[PreTrainedModel]]:
    """
    Add Gist special tokens to tokenizer and resize model embeddings.

    Args:
        tokenizer: Hugging Face tokenizer
        model: Hugging Face model
        num_gist_tokens: Number of Gist tokens to add

    Returns:
        Tuple of (modified tokenizer, modified model)
    """
    # Generate Gist token strings
    gist_tokens = [f"<GIST_{i}>" for i in range(num_gist_tokens)]

    # Check if tokens already exist (idempotency)
    existing_tokens = tokenizer.get_vocab()
    tokens_to_add = [token for token in gist_tokens if token not in existing_tokens]

    if not tokens_to_add:
        # All tokens already exist
        return tokenizer, model

    # Add tokens to tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    # Resize model embedding layer (if model is provided)
    if model is not None:
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def get_gist_token_ids(tokenizer: PreTrainedTokenizer, num_gist_tokens: int) -> list[int]:
    """
    Get token IDs for Gist tokens.

    Args:
        tokenizer: Tokenizer with Gist tokens added
        num_gist_tokens: Number of Gist tokens

    Returns:
        List of Gist token IDs
    """
    return [
        tokenizer.convert_tokens_to_ids(f"<GIST_{i}>")
        for i in range(num_gist_tokens)
    ]
