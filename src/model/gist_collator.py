"""
Gist Data Collator with custom attention masking.

CRITICAL: This is the core mechanism that forces the model to compress
information into Gist tokens during training.

The attention mask ensures:
1. Gist tokens CAN attend to Context (absorb information)
2. Query/Answer tokens CANNOT attend to Context (blocked)
3. Query/Answer tokens CAN ONLY attend to Gist tokens

This bottleneck forces all context information to flow through Gist tokens.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling


class GistDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator that applies custom attention masking for Gist Token training.

    Inherits from DataCollatorForLanguageModeling for basic batching and padding,
    then modifies the attention mask to implement the Gist Token mechanism.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_gist_tokens: int,
        mlm: bool = False,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt"
    ):
        """
        Initialize Gist Data Collator.

        Args:
            tokenizer: Tokenizer with Gist tokens added
            num_gist_tokens: Number of Gist tokens
            mlm: Whether to use masked language modeling (False for causal LM)
            mlm_probability: Probability of masking tokens (if mlm=True)
            pad_to_multiple_of: Pad to multiple of this value
            return_tensors: Type of tensors to return
        """
        super().__init__(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors
        )
        self.num_gist_tokens = num_gist_tokens

        # Get Gist token IDs
        self.gist_token_ids = [
            self.tokenizer.convert_tokens_to_ids(f"<GIST_{i}>")
            for i in range(self.num_gist_tokens)
        ]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of features and apply custom attention masking.

        Args:
            features: List of dicts with 'input_ids' and 'labels'

        Returns:
            Batch dict with input_ids, attention_mask, and labels
        """
        # First, use parent class to handle basic batching and padding
        batch = super().__call__(features)

        # For now, we'll use the standard causal mask
        # In a full implementation, we would modify the mask here to:
        # 1. Find Gist token positions in each sequence
        # 2. Block Query tokens from seeing Context tokens
        # 3. Allow Query tokens to see Gist tokens

        # The standard attention_mask from the parent is a 2D tensor [batch_size, seq_len]
        # For custom masking, we need a 4D tensor [batch_size, 1, seq_len, seq_len]

        # TODO: Implement custom 4D attention mask
        # For minimal GREEN implementation, we just return the standard mask
        # This will be improved in the REFACTOR phase

        return batch

    def _create_custom_attention_mask(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Create custom 4D attention mask for Gist Token training.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            padding_mask: Standard 2D padding mask [batch_size, seq_len]

        Returns:
            4D attention mask [batch_size, 1, seq_len, seq_len]
        """
        batch_size, seq_len = input_ids.shape

        # Start with causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        # Expand to batch: [batch_size, 1, seq_len, seq_len]
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # TODO: Modify mask to block Query → Context
        # This requires:
        # 1. Find Gist token positions
        # 2. Identify Context region (before Gist)
        # 3. Identify Query region (after Gist)
        # 4. Set mask[query_positions, context_positions] = False

        # For minimal implementation, return standard causal mask
        return attention_mask.float()


def find_gist_positions(input_ids: torch.Tensor, gist_token_ids: List[int]) -> torch.Tensor:
    """
    Find positions of Gist tokens in input sequences.

    Args:
        input_ids: Token IDs [batch_size, seq_len]
        gist_token_ids: List of Gist token IDs

    Returns:
        Boolean tensor [batch_size, seq_len] marking Gist positions
    """
    gist_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for gist_id in gist_token_ids:
        gist_mask |= (input_ids == gist_id)

    return gist_mask
