"""
Unit tests for NIAH (Needle in Haystack) generator.

Tests ensure that synthetic datasets are properly generated with embedded needles
for evaluating compression algorithms' information retention.
"""

import pytest
from pathlib import Path
import json


class TestNIAHGenerator:
    """Test suite for NIAH dataset generator."""

    def test_generator_initialization(self):
        """Test that NIAH generator can be initialized."""
        from src.data.create_niah import NIAHGenerator

        generator = NIAHGenerator(
            output_dir="data/processed/niah",
            num_samples=10,
            context_length_range=(2000, 8000)
        )

        assert generator.output_dir == Path("data/processed/niah")
        assert generator.num_samples == 10
        assert generator.context_length_range == (2000, 8000)

    def test_generate_background_text(self):
        """Test generation of background text with target length."""
        from src.data.create_niah import NIAHGenerator

        generator = NIAHGenerator("data/processed/niah", num_samples=1)

        # Generate background of ~4000 tokens (rough estimation: 4 chars/token)
        background = generator.generate_background_text(target_chars=16000)

        assert len(background) >= 15000, "Background too short"
        assert len(background) <= 17000, "Background too long"
        assert isinstance(background, str)

    def test_generate_needle(self):
        """Test generation of random needle (secret information)."""
        from src.data.create_niah import NIAHGenerator

        generator = NIAHGenerator("data/processed/niah", num_samples=1)

        needle = generator.generate_needle()

        # Should be a passkey format like "X7G9K2" or similar
        assert isinstance(needle, str)
        assert len(needle) >= 4, "Needle too short"
        assert len(needle) <= 10, "Needle too long"

    def test_insert_needle_in_context(self):
        """Test inserting needle at specific position in context."""
        from src.data.create_niah import NIAHGenerator

        generator = NIAHGenerator("data/processed/niah", num_samples=1)

        context = "A" * 1000
        needle = "SECRET123"
        position = 0.5  # Middle

        result = generator.insert_needle(context, needle, position)

        assert needle in result
        assert len(result) > len(context)
        # Verify needle is roughly in the middle
        needle_pos = result.index(needle)
        assert 400 < needle_pos < 600, "Needle not in expected position"

    def test_generate_single_sample(self):
        """Test generation of a complete NIAH sample."""
        from src.data.create_niah import NIAHGenerator

        generator = NIAHGenerator("data/processed/niah", num_samples=1)

        sample = generator.generate_sample(
            target_length=4000,
            needle_position=0.5
        )

        # Verify required fields
        assert "context" in sample
        assert "question" in sample
        assert "answer" in sample
        assert "needle" in sample
        assert "needle_position" in sample

        # Verify types
        assert isinstance(sample["context"], str)
        assert isinstance(sample["answer"], str)
        assert sample["needle"] in sample["context"]

        # Verify question asks about the needle
        assert "passkey" in sample["question"].lower() or \
               "secret" in sample["question"].lower() or \
               "password" in sample["question"].lower()

    def test_generate_multiple_samples(self):
        """Test batch generation with varying positions."""
        from src.data.create_niah import NIAHGenerator

        generator = NIAHGenerator(
            output_dir="data/processed/niah",
            num_samples=5
        )

        samples = generator.generate_all()

        assert len(samples) == 5

        # Check diversity of needle positions
        positions = [s["needle_position"] for s in samples]
        assert len(set(positions)) >= 3, "Not enough position diversity"

        # Verify all samples have required fields
        for sample in samples:
            assert all(k in sample for k in ["context", "question", "answer", "needle"])

    def test_save_to_jsonl(self):
        """Test saving samples to JSONL format."""
        from src.data.create_niah import NIAHGenerator
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = NIAHGenerator(
                output_dir=tmpdir,
                num_samples=3
            )

            samples = generator.generate_all()
            output_file = generator.save_jsonl(samples)

            # Verify file exists
            assert output_file.exists()

            # Verify JSONL format
            with open(output_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3

                for line in lines:
                    data = json.loads(line)
                    assert "context" in data
                    assert "needle" in data

    def test_context_length_variation(self):
        """Test that generated contexts vary in length."""
        from src.data.create_niah import NIAHGenerator

        generator = NIAHGenerator(
            output_dir="data/processed/niah",
            num_samples=10,
            context_length_range=(2000, 8000)
        )

        samples = generator.generate_all()

        lengths = [len(s["context"]) for s in samples]

        # Should have variation (not all same length)
        assert len(set(lengths)) >= 5, "Not enough length diversity"

        # Should be within specified range (chars ~= 4 * tokens)
        min_chars = 2000 * 4 * 0.9  # Allow 10% tolerance
        max_chars = 8000 * 4 * 1.1

        for length in lengths:
            assert min_chars <= length <= max_chars, f"Length {length} out of range"

    def test_needle_format_consistency(self):
        """Test that all needles follow consistent format."""
        from src.data.create_niah import NIAHGenerator

        generator = NIAHGenerator("data/processed/niah", num_samples=10)

        samples = generator.generate_all()

        for sample in samples:
            needle = sample["needle"]
            # Should be alphanumeric
            assert needle.replace("-", "").isalnum(), f"Invalid needle format: {needle}"
