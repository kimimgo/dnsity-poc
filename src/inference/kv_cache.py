"""
KV Cache compression and serialization for Gist Tokens.

Provides functions to:
1. Extract Gist token KV from past_key_values
2. Save/load compressed KV to/from disk (.safetensors)
3. Inject saved KV for efficient inference
"""

from typing import Tuple, Dict, Any, Optional
import torch
from torch import Tensor
from pathlib import Path
from transformers import PreTrainedModel, PreTrainedTokenizer
from safetensors.torch import save_file, load_file
import json


def extract_gist_kv(
    past_key_values: Tuple[Tuple[Tensor, Tensor], ...],
    gist_start: int,
    gist_end: int
) -> Tuple[Tuple[Tensor, Tensor], ...]:
    """
    Extract only Gist token KV from full past_key_values.

    Args:
        past_key_values: Full KV cache from model forward pass
            Format: Tuple[Tuple[key, value], ...] for each layer
            Shape: (batch, num_heads, seq_len, head_dim)
        gist_start: Starting position of Gist tokens
        gist_end: Ending position of Gist tokens (exclusive)

    Returns:
        Compressed KV cache with only Gist tokens
    """
    gist_kv = tuple(
        (
            layer_kv[0][:, :, gist_start:gist_end, :].contiguous(),  # Key
            layer_kv[1][:, :, gist_start:gist_end, :].contiguous()   # Value
        )
        for layer_kv in past_key_values
    )

    return gist_kv


def detect_gist_positions(
    input_ids: Tensor,
    tokenizer: PreTrainedTokenizer
) -> Tuple[Optional[int], Optional[int]]:
    """
    Detect Gist token positions in input_ids.

    Args:
        input_ids: Token IDs tensor (batch, seq_len)
        tokenizer: Tokenizer with Gist tokens added

    Returns:
        (gist_start, gist_end) positions, or (None, None) if not found
    """
    # Get Gist token IDs
    gist_token_ids = []
    vocab = tokenizer.get_vocab()

    for token, token_id in vocab.items():
        if token.startswith("<GIST_"):
            gist_token_ids.append(token_id)

    if not gist_token_ids:
        return None, None

    # Find first Gist token
    input_ids_list = input_ids[0].tolist()
    gist_start = None

    for i, token_id in enumerate(input_ids_list):
        if token_id in gist_token_ids:
            gist_start = i
            break

    if gist_start is None:
        return None, None

    # Find end of Gist tokens (consecutive)
    gist_end = gist_start
    for i in range(gist_start, len(input_ids_list)):
        if input_ids_list[i] in gist_token_ids:
            gist_end = i + 1
        else:
            break

    return gist_start, gist_end


def save_gist_kv(
    gist_kv: Tuple[Tuple[Tensor, Tensor], ...],
    save_path: Path | str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save Gist KV cache to .safetensors format.

    Args:
        gist_kv: Compressed KV cache
        save_path: Path to save file
        metadata: Optional metadata dict (num_gist_tokens, model_name, etc.)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten KV cache to dict
    tensors = {}
    for layer_idx, (key, value) in enumerate(gist_kv):
        tensors[f"layer_{layer_idx}_key"] = key
        tensors[f"layer_{layer_idx}_value"] = value

    # Convert metadata values to strings for safetensors
    str_metadata = {}
    if metadata:
        for k, v in metadata.items():
            str_metadata[k] = str(v)

    # Save tensors
    save_file(tensors, str(save_path), metadata=str_metadata if str_metadata else None)

    # Save metadata as JSON sidecar (with original types)
    if metadata:
        metadata_path = save_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_gist_kv(
    load_path: Path | str
) -> Tuple[Tuple[Tuple[Tensor, Tensor], ...], Dict[str, Any]]:
    """
    Load Gist KV cache from .safetensors format.

    Args:
        load_path: Path to saved file

    Returns:
        (gist_kv, metadata) tuple
    """
    load_path = Path(load_path)

    # Load tensors
    tensors = load_file(str(load_path))

    # Load metadata
    metadata_path = load_path.with_suffix(".json")
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Reconstruct KV cache structure
    num_layers = metadata.get("num_layers", len(tensors) // 2)
    gist_kv = tuple(
        (
            tensors[f"layer_{layer_idx}_key"],
            tensors[f"layer_{layer_idx}_value"]
        )
        for layer_idx in range(num_layers)
    )

    return gist_kv, metadata


def inject_gist_kv(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    gist_kv: Tuple[Tuple[Tensor, Tensor], ...],
    question: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Inject Gist KV cache and generate answer to question.

    Args:
        model: Pre-trained model
        tokenizer: Tokenizer
        gist_kv: Compressed KV cache to inject
        question: Question to answer
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        Dict with 'generated_text', 'input_length', 'output_length'
    """
    # Tokenize question
    inputs = tokenizer(question, return_tensors="pt")

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Move gist_kv to device
    gist_kv = tuple(
        (key.to(device), value.to(device))
        for key, value in gist_kv
    )

    # For newer transformers versions, we need to prepend dummy tokens
    # to account for the Gist KV cache length
    gist_len = gist_kv[0][0].shape[2]  # seq_len dimension

    # Create dummy input_ids to represent Gist tokens in cache
    dummy_gist_ids = torch.zeros(
        (inputs["input_ids"].shape[0], gist_len),
        dtype=torch.long,
        device=device
    )

    # Prepend dummy IDs and adjust attention mask
    full_input_ids = torch.cat([dummy_gist_ids, inputs["input_ids"]], dim=1)
    full_attention_mask = torch.ones_like(full_input_ids)

    # Generate with injected KV cache
    with torch.no_grad():
        outputs = model.generate(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            past_key_values=gist_kv,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True
        )

    # Decode (skip the dummy + question part)
    skip_len = full_input_ids.shape[1]
    generated_text = tokenizer.decode(
        outputs[0][skip_len:],
        skip_special_tokens=True
    )

    return {
        "generated_text": generated_text.strip(),
        "input_length": inputs["input_ids"].shape[1],
        "output_length": outputs.shape[1] - skip_len
    }
