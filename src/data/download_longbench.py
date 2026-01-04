"""
LongBench dataset downloader.

Downloads and processes LongBench benchmark tasks for Gist Token evaluation.
Uses direct file download from Hugging Face Hub to avoid datasets library compatibility issues.
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
import zipfile
import tempfile
from huggingface_hub import hf_hub_download


class LongBenchDownloader:
    """
    Downloader for LongBench benchmark datasets.

    Supports downloading specific tasks and formatting them for Gist Token experiments.
    """

    VALID_TASKS = [
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "multifieldqa_zh",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "dureader",
        "gov_report",
        "qmsum",
        "multi_news",
        "vcsum",
        "trec",
        "triviaqa",
        "samsum",
        "lsht",
        "passage_count",
        "passage_retrieval_en",
        "passage_retrieval_zh",
        "lcc",
        "repobench-p",
    ]

    def __init__(self, output_dir: str, tasks: List[str]):
        """
        Initialize the downloader.

        Args:
            output_dir: Directory to save processed datasets
            tasks: List of task names to download
        """
        self.output_dir = Path(output_dir)
        self.tasks = tasks
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._data_cache = None  # Cache for extracted data

    def _download_and_extract_data(self) -> Path:
        """
        Download and extract the LongBench data.zip file.

        Returns:
            Path to the extracted data directory
        """
        if self._data_cache and self._data_cache.exists():
            return self._data_cache

        # Download data.zip from Hugging Face Hub
        print("Downloading LongBench data.zip from Hugging Face...")
        zip_path = hf_hub_download(
            repo_id="THUDM/LongBench",
            filename="data.zip",
            repo_type="dataset",
            cache_dir=str(self.output_dir.parent / ".cache")
        )

        # Extract to temp directory
        extract_dir = self.output_dir.parent / ".longbench_data"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        self._data_cache = extract_dir / "data"
        return self._data_cache

    def download_task(self, task_name: str, max_samples: Optional[int] = None) -> Dict:
        """
        Download a single LongBench task.

        Args:
            task_name: Name of the task to download
            max_samples: Maximum number of samples to download (None for all)

        Returns:
            Dictionary with download results

        Raises:
            ValueError: If task_name is not valid
        """
        if task_name not in self.VALID_TASKS:
            raise ValueError(f"Invalid task: {task_name}. Must be one of {self.VALID_TASKS}")

        # Download and extract data if needed
        data_dir = self._download_and_extract_data()

        # Find the task file (jsonl format)
        task_file = data_dir / f"{task_name}.jsonl"
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")

        # Load samples from JSONL
        samples = []
        with open(task_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        # Limit samples if specified
        if max_samples is not None:
            samples = samples[:max_samples]

        # Format samples to our standard format
        formatted_samples = []
        for item in samples:
            sample = {
                "context": item.get("context", ""),
                "question": item.get("input", ""),
                "answer": item.get("answers", [""])[0] if isinstance(item.get("answers"), list) else item.get("answers", ""),
                "task": task_name
            }
            formatted_samples.append(sample)

        return {
            "task": task_name,
            "count": len(formatted_samples),
            "success": True,
            "data": formatted_samples
        }

    def download_and_save(self, task_name: str, max_samples: Optional[int] = None) -> Path:
        """
        Download a task and save it to JSONL format.

        Args:
            task_name: Name of the task to download
            max_samples: Maximum number of samples to download

        Returns:
            Path to the saved JSONL file
        """
        result = self.download_task(task_name, max_samples)

        # Save to JSONL
        output_file = self.output_dir / f"{task_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in result["data"]:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        return output_file

    def download_all(self, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Download all configured tasks.

        Args:
            max_samples: Maximum number of samples per task

        Returns:
            List of download results for each task
        """
        results = []
        for task in self.tasks:
            try:
                result = self.download_task(task, max_samples)
                results.append(result)
                print(f"✅ {task}: {result['count']} samples")
            except Exception as e:
                print(f"❌ {task}: {str(e)}")
                results.append({
                    "task": task,
                    "count": 0,
                    "success": False,
                    "error": str(e)
                })

        return results


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Download LongBench datasets")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["narrativeqa", "qasper", "gov_report"],
        help="Tasks to download"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/longbench",
        help="Output directory"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples per task"
    )

    args = parser.parse_args()

    downloader = LongBenchDownloader(
        output_dir=args.output_dir,
        tasks=args.tasks
    )

    print(f"Downloading {len(args.tasks)} tasks...")
    results = downloader.download_all(max_samples=args.max_samples)

    # Save all tasks to JSONL
    for task in args.tasks:
        downloader.download_and_save(task, max_samples=args.max_samples)

    print(f"\n✅ Downloaded {sum(r['count'] for r in results)} total samples")
    print(f"📁 Saved to {downloader.output_dir}")


if __name__ == "__main__":
    main()
