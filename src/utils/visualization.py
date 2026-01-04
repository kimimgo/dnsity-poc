"""
Attention mask visualization utilities.

Provides functions to visualize attention masks for debugging Gist Token attention masking.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional


def visualize_attention_mask(
    mask: torch.Tensor,
    positions: Optional[dict] = None,
    title: str = "Attention Mask",
    output_path: str = "attention_mask.png"
) -> None:
    """
    Visualize attention mask as a heatmap.

    Args:
        mask: Attention mask tensor (2D or 4D)
              - 2D: [seq_len, seq_len]
              - 4D: [batch, heads, seq_len, seq_len]
        positions: Optional dict with region boundaries:
                   - context_end: End index of context region
                   - gist_start: Start index of gist region
                   - gist_end: End index of gist region
                   - query_start: Start index of query region
        title: Plot title
        output_path: Where to save the PNG file
    """
    # Convert to numpy and handle different dimensions
    if mask.dim() == 4:
        # 4D mask: [batch, heads, seq_len, seq_len]
        # Take first batch, first head
        mask_2d = mask[0, 0].detach().cpu().numpy()
    elif mask.dim() == 2:
        # 2D mask: [seq_len, seq_len]
        mask_2d = mask.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported mask dimension: {mask.dim()}. Expected 2D or 4D.")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(mask_2d, cmap='Blues', aspect='auto', interpolation='nearest')

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Attention Allowed')

    # Add region labels if positions are provided
    if positions is not None:
        context_end = positions.get("context_end", 0)
        gist_start = positions.get("gist_start", 0)
        gist_end = positions.get("gist_end", 0)
        query_start = positions.get("query_start", 0)

        # Draw vertical lines to separate regions
        if context_end > 0:
            ax.axvline(x=context_end - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        if gist_start > 0:
            ax.axvline(x=gist_start - 0.5, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        if gist_end > 0:
            ax.axvline(x=gist_end - 0.5, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        if query_start > 0:
            ax.axvline(x=query_start - 0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

        # Draw horizontal lines
        if context_end > 0:
            ax.axhline(y=context_end - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        if gist_start > 0:
            ax.axhline(y=gist_start - 0.5, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        if gist_end > 0:
            ax.axhline(y=gist_end - 0.5, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        if query_start > 0:
            ax.axhline(y=query_start - 0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

        # Add region labels
        seq_len = mask_2d.shape[0]
        label_y = -0.5

        if context_end > 0:
            ax.text(context_end / 2, label_y, 'Context', ha='center', va='top',
                   fontsize=10, color='red', weight='bold')
        if gist_start < gist_end:
            ax.text((gist_start + gist_end) / 2, label_y, 'Gist', ha='center', va='top',
                   fontsize=10, color='green', weight='bold')
        if query_start < seq_len:
            ax.text((query_start + seq_len) / 2, label_y, 'Query', ha='center', va='top',
                   fontsize=10, color='orange', weight='bold')

    # Set labels and title
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
