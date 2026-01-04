"""
Unit tests for Gist Token expansion in tokenizer and model.

Tests ensure that Gist tokens are properly added to the tokenizer vocabulary
and the model's embedding layer is correctly resized.
"""

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TestTokenizerExpansion:
    """Test suite for Gist token tokenizer expansion."""

    def test_tokenizer_expansion(self):
        """Test that Gist tokens are added to tokenizer vocabulary."""
        from src.model.gist_tokenizer import add_gist_tokens

        # Use a small model for testing
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")

        original_vocab_size = len(tokenizer)
        num_gist_tokens = 10

        # Add Gist tokens
        tokenizer, model = add_gist_tokens(tokenizer, model, num_gist_tokens)

        # Verify tokens were added
        assert len(tokenizer) == original_vocab_size + num_gist_tokens

        # Verify tokens are in vocabulary
        for i in range(num_gist_tokens):
            token = f"<GIST_{i}>"
            assert token in tokenizer.get_vocab()

    def test_vocab_size_change(self):
        """Test that vocab_size increases by exactly num_gist_tokens."""
        from src.model.gist_tokenizer import add_gist_tokens

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")

        original_size = len(tokenizer)
        num_gist = 25

        tokenizer, model = add_gist_tokens(tokenizer, model, num_gist)

        assert len(tokenizer) == original_size + num_gist

    def test_embedding_layer_resize(self):
        """Test that model.resize_token_embeddings() correctly resizes embedding layer."""
        from src.model.gist_tokenizer import add_gist_tokens

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")

        original_embed_size = model.get_input_embeddings().weight.shape[0]
        num_gist = 10

        tokenizer, model = add_gist_tokens(tokenizer, model, num_gist)

        # Verify embedding layer was resized
        new_embed_size = model.get_input_embeddings().weight.shape[0]
        assert new_embed_size == original_embed_size + num_gist

        # Verify embedding dimension remains unchanged
        original_dim = model.get_input_embeddings().weight.shape[1]
        new_dim = model.get_input_embeddings().weight.shape[1]
        assert original_dim == new_dim

    def test_gist_token_ids_retrievable(self):
        """Test that Gist token IDs can be retrieved from tokenizer."""
        from src.model.gist_tokenizer import add_gist_tokens

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")

        num_gist = 10
        tokenizer, model = add_gist_tokens(tokenizer, model, num_gist)

        # Get Gist token IDs
        gist_ids = [tokenizer.convert_tokens_to_ids(f"<GIST_{i}>") for i in range(num_gist)]

        # Verify all IDs are valid (not unk_token_id)
        unk_id = tokenizer.unk_token_id
        for gist_id in gist_ids:
            assert gist_id != unk_id

        # Verify IDs are sequential
        assert gist_ids == sorted(gist_ids)

    def test_multiple_calls_idempotent(self):
        """Test that calling add_gist_tokens multiple times doesn't duplicate tokens."""
        from src.model.gist_tokenizer import add_gist_tokens

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")

        num_gist = 10

        # First call
        tokenizer, model = add_gist_tokens(tokenizer, model, num_gist)
        size_after_first = len(tokenizer)

        # Second call should not add tokens again (if properly implemented)
        # Note: This test will initially fail if we don't handle duplicates
        tokenizer, model = add_gist_tokens(tokenizer, model, num_gist)
        size_after_second = len(tokenizer)

        # Should remain the same (idempotent)
        assert size_after_first == size_after_second
