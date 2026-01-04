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

        # Create custom 4D attention mask for Gist Token training
        # This forces Question tokens to ONLY see Gist tokens, not Context
        input_ids = batch["input_ids"]
        padding_mask = batch.get("attention_mask")

        # Create custom mask
        custom_mask = self._create_custom_attention_mask(
            input_ids=input_ids,
            padding_mask=padding_mask
        )

        # Replace standard 2D mask with custom 4D mask
        batch["attention_mask"] = custom_mask

        return batch

    def _create_custom_attention_mask(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Create custom 4D attention mask for Gist Token training.

        Masking strategy:
        - Context tokens can see all previous context
        - Gist tokens can see all context (to absorb information)
        - Question/Answer tokens CANNOT see context directly
        - Question/Answer tokens CAN ONLY see Gist tokens

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            padding_mask: Standard 2D padding mask [batch_size, seq_len]

        Returns:
            4D attention mask [batch_size, 1, seq_len, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Start with causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

        # Expand to batch: [batch_size, 1, seq_len, seq_len]
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).clone()

        # Process each sample in batch
        for batch_idx in range(batch_size):
            sample_ids = input_ids[batch_idx]

            # Find Gist token positions
            gist_start = None
            gist_end = None

            for i, token_id in enumerate(sample_ids):
                if token_id.item() in self.gist_token_ids:
                    if gist_start is None:
                        gist_start = i
                    gist_end = i + 1

            # If no Gist tokens found, skip custom masking for this sample
            if gist_start is None:
                continue

            # Context region: [0:gist_start]
            # Gist region: [gist_start:gist_end]
            # Question/Answer region: [gist_end:seq_len]

            # CRITICAL: Block Question/Answer from seeing Context
            # For all positions after Gist (Question/Answer region),
            # block attention to Context region (before Gist)
            if gist_end < seq_len:
                # attention_mask[batch_idx, 0, query_pos, context_pos] = False
                attention_mask[batch_idx, 0, gist_end:, :gist_start] = False

                # Ensure Question/Answer can still see Gist tokens
                # (This is already True from causal mask, just being explicit)
                attention_mask[batch_idx, 0, gist_end:, gist_start:gist_end] = True

        # Apply padding mask if provided
        if padding_mask is not None:
            # Expand padding mask to 4D: [batch, 1, 1, seq_len]
            padding_mask_4d = padding_mask.unsqueeze(1).unsqueeze(1)

            # Mask out padded positions (both rows and columns)
            attention_mask = attention_mask & padding_mask_4d
            attention_mask = attention_mask & padding_mask.unsqueeze(1).unsqueeze(2)

        # Convert to float (transformers expects float attention masks)
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
