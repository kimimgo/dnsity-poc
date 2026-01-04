"""
Full Context Baseline implementation.

Uses the full context without compression for comparison with Gist Token approach.
"""

from typing import Optional, Callable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class FullContextBaseline:
    """
    Baseline that uses full context without compression.
    
    This serves as an upper bound for quality (all information preserved)
    and a comparison point for VRAM usage.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        load_in_4bit: bool = False,
        device: str = "auto"
    ):
        """
        Initialize Full Context baseline.
        
        Args:
            model_name: Hugging Face model name
            load_in_4bit: Whether to use 4-bit quantization
            device: Device to load model on ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        
        # Load model
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            if device != "auto" and torch.cuda.is_available():
                self.model = self.model.to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def inference(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Perform inference with full context.
        
        Args:
            context: Full context text
            question: Question to answer
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        
        Returns:
            Generated answer text
        """
        # Create prompt
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Limit to avoid OOM
        )
        
        # Move to same device as model
        if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode only the new tokens
        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return answer.strip()


def measure_vram_usage(func: Callable) -> float:
    """
    Measure peak VRAM usage of a function.
    
    Args:
        func: Function to measure
    
    Returns:
        Peak VRAM usage in GB
    """
    if not torch.cuda.is_available():
        return 0.0
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Run function
    func()
    
    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated()
    peak_memory_gb = peak_memory / 1e9
    
    return peak_memory_gb
