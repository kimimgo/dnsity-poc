"""
Unit tests for LongBench dataset download functionality.

Tests ensure that datasets are properly downloaded, filtered, and formatted.
"""

import pytest
from pathlib import Path
import json


class TestLongBenchDownloader:
    """Test suite for LongBench dataset downloader."""

    def test_downloader_initialization(self):
        """Test that downloader can be initialized with correct parameters."""
        from src.data.download_longbench import LongBenchDownloader

        downloader = LongBenchDownloader(
            output_dir="data/processed/longbench",
            tasks=["narrativeqa", "qasper", "gov_report"]
        )

        assert downloader.output_dir == Path("data/processed/longbench")
        assert len(downloader.tasks) == 3
        assert "narrativeqa" in downloader.tasks

    def test_download_single_task(self):
        """Test downloading a single task with sample limit."""
        from src.data.download_longbench import LongBenchDownloader

        downloader = LongBenchDownloader(
            output_dir="data/processed/longbench",
            tasks=["narrativeqa"]
        )

        # Download with small sample for testing
        result = downloader.download_task("narrativeqa", max_samples=5)

        assert result["task"] == "narrativeqa"
        assert result["count"] == 5
        assert result["success"] is True

    def test_sample_format_validation(self):
        """Test that downloaded samples have required fields."""
        from src.data.download_longbench import LongBenchDownloader

        downloader = LongBenchDownloader(
            output_dir="data/processed/longbench",
            tasks=["narrativeqa"]
        )

        samples = downloader.download_task("narrativeqa", max_samples=1)

        # Check required fields
        sample = samples["data"][0]
        assert "context" in sample
        assert "question" in sample
        assert "answer" in sample
        assert "task" in sample

        # Validate types
        assert isinstance(sample["context"], str)
        assert isinstance(sample["question"], str)
        assert len(sample["context"]) > 0

    def test_save_to_jsonl(self):
        """Test saving dataset to JSONL format."""
        from src.data.download_longbench import LongBenchDownloader
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = LongBenchDownloader(
                output_dir=tmpdir,
                tasks=["narrativeqa"]
            )

            # Download and save
            downloader.download_and_save("narrativeqa", max_samples=3)

            # Verify file exists
            output_file = Path(tmpdir) / "narrativeqa.jsonl"
            assert output_file.exists()

            # Verify JSONL format
            with open(output_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3

                for line in lines:
                    data = json.loads(line)
                    assert "context" in data
                    assert "question" in data

    def test_token_count_estimation(self):
        """Test that contexts have sufficient length for compression experiments."""
        from src.data.download_longbench import LongBenchDownloader

        downloader = LongBenchDownloader(
            output_dir="data/processed/longbench",
            tasks=["gov_report"]
        )

        samples = downloader.download_task("gov_report", max_samples=5)

        for sample in samples["data"]:
            # Rough estimation: ~4 chars per token
            estimated_tokens = len(sample["context"]) / 4
            # LongBench is designed for long contexts - verify minimum length only
            assert estimated_tokens >= 1000, "Context too short for compression testing"
            # No upper bound - LongBench intentionally includes very long docs

    def test_invalid_task_name(self):
        """Test that invalid task names raise appropriate errors."""
        from src.data.download_longbench import LongBenchDownloader

        downloader = LongBenchDownloader(
            output_dir="data/processed/longbench",
            tasks=["invalid_task"]
        )

        with pytest.raises(ValueError, match="Invalid task"):
            downloader.download_task("invalid_task")

    def test_download_all_tasks(self):
        """Test downloading multiple tasks in batch."""
        from src.data.download_longbench import LongBenchDownloader

        downloader = LongBenchDownloader(
            output_dir="data/processed/longbench",
            tasks=["narrativeqa", "qasper", "gov_report"]
        )

        results = downloader.download_all(max_samples=5)

        assert len(results) == 3
        assert all(r["success"] for r in results)
        assert sum(r["count"] for r in results) == 15  # 3 tasks × 5 samples
