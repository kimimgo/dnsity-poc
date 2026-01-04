"""
Unit tests for attention mask visualization.

These tests verify that visualization functions can generate correct plots
for debugging Gist Token attention masking.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile


class TestAttentionVisualization:
    """Test suite for attention mask visualization."""

    def test_visualize_attention_mask_creates_file(self):
        """Test that visualization creates an output file."""
        from src.utils.visualization import visualize_attention_mask

        # Create a simple attention mask
        mask = torch.tril(torch.ones(10, 10))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_mask.png"

            visualize_attention_mask(
                mask=mask,
                output_path=str(output_path)
            )

            assert output_path.exists(), "Visualization should create output file"

    def test_visualize_with_labels(self):
        """Test visualization with Context/Gist/Query labels."""
        from src.utils.visualization import visualize_attention_mask

        mask = torch.tril(torch.ones(10, 10))

        # Define regions
        positions = {
            "context_end": 5,
            "gist_start": 5,
            "gist_end": 7,
            "query_start": 7
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_labeled.png"

            visualize_attention_mask(
                mask=mask,
                positions=positions,
                output_path=str(output_path)
            )

            assert output_path.exists()

    def test_visualize_handles_4d_mask(self):
        """Test that visualization handles 4D masks [batch, heads, seq, seq]."""
        from src.utils.visualization import visualize_attention_mask

        # 4D mask: [batch=1, heads=1, seq=8, seq=8]
        mask = torch.tril(torch.ones(1, 1, 8, 8))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_4d.png"

            visualize_attention_mask(
                mask=mask,
                output_path=str(output_path)
            )

            assert output_path.exists()

    def test_visualize_handles_2d_mask(self):
        """Test that visualization handles 2D masks [seq, seq]."""
        from src.utils.visualization import visualize_attention_mask

        mask = torch.tril(torch.ones(8, 8))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_2d.png"

            visualize_attention_mask(
                mask=mask,
                output_path=str(output_path)
            )

            assert output_path.exists()

    def test_visualize_custom_title(self):
        """Test visualization with custom title."""
        from src.utils.visualization import visualize_attention_mask

        mask = torch.ones(6, 6)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_title.png"

            visualize_attention_mask(
                mask=mask,
                title="Custom Attention Mask",
                output_path=str(output_path)
            )

            assert output_path.exists()
