"""
Evaluation metrics for Gist Token experiments.

Provides functions to calculate:
- Passkey retrieval accuracy
- Compression ratio
- VRAM usage
- Throughput
"""

import re
import time
import torch
from typing import List, Dict, Callable, Optional


def extract_passkey(text: str, pattern: str = r'\d{5,}') -> Optional[str]:
    """
    Extract passkey from generated text using regex.
    
    Args:
        text: Generated text containing passkey
        pattern: Regex pattern to match passkey
    
    Returns:
        Extracted passkey or None if not found
    """
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return None


def calculate_passkey_accuracy(results: List[Dict[str, str]]) -> float:
    """
    Calculate passkey retrieval accuracy.
    
    Args:
        results: List of dicts with 'predicted' and 'ground_truth' keys
    
    Returns:
        Accuracy as float (0.0 to 1.0)
    """
    if not results:
        return 0.0
    
    correct = sum(
        1 for r in results 
        if r.get("predicted") == r.get("ground_truth")
    )
    
    return correct / len(results)


def calculate_compression_ratio(
    original_length: int,
    compressed_length: int
) -> float:
    """
    Calculate compression ratio.
    
    Args:
        original_length: Original context length in tokens
        compressed_length: Number of Gist tokens
    
    Returns:
        Compression ratio (e.g., 400.0 for 4000→10)
    """
    if compressed_length == 0:
        return float('inf')
    
    return original_length / compressed_length


def measure_vram_mb(func: Callable, reset: bool = True) -> float:
    """
    Measure peak VRAM usage of a function in MB.
    
    Args:
        func: Function to measure
        reset: Whether to reset peak memory stats before measuring
    
    Returns:
        Peak VRAM usage in MB
    """
    if not torch.cuda.is_available():
        return 0.0
    
    if reset:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Run function
    func()
    
    # Get peak memory in MB
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    
    return peak_memory_mb


def calculate_throughput(num_tokens: int, elapsed_time: float) -> float:
    """
    Calculate generation throughput.
    
    Args:
        num_tokens: Number of tokens generated
        elapsed_time: Time elapsed in seconds
    
    Returns:
        Throughput in tokens/second
    """
    if elapsed_time == 0:
        return 0.0
    
    return num_tokens / elapsed_time


def measure_inference_time(func: Callable) -> float:
    """
    Measure inference time of a function.
    
    Args:
        func: Function to measure
    
    Returns:
        Elapsed time in seconds
    """
    start_time = time.time()
    func()
    end_time = time.time()
    
    return end_time - start_time
