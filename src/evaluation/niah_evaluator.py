"""
NIAH (Needle in a Haystack) Evaluator.

Evaluates passkey retrieval accuracy for Gist Token compression.
"""

from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluation.metrics import extract_passkey


class NIAHEvaluator:
    """
    Evaluator for Needle-in-a-Haystack (Passkey Retrieval) tasks.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "auto"
    ):
        """
        Initialize NIAH evaluator.
        
        Args:
            model_name: Model for evaluation
            device: Device to run on
        """
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move to device
        if device != "auto" and torch.cuda.is_available():
            self.model = self.model.to(device)
    
    def evaluate_sample(
        self,
        sample: Dict[str, str],
        max_new_tokens: int = 50,
        temperature: float = 0.7
    ) -> Dict[str, str]:
        """
        Evaluate a single NIAH sample.
        
        Args:
            sample: Dict with 'context', 'question', 'answer' keys
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dict with 'predicted' and 'ground_truth' keys
        """
        # Build prompt
        prompt = f"{sample['context']}\n\nQuestion: {sample['question']}\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        # Move to same device as model
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract passkey
        predicted = extract_passkey(generated_text)
        
        return {
            "predicted": predicted if predicted else generated_text.strip()[:50],
            "ground_truth": sample["answer"],
            "generated_full": generated_text.strip()
        }
    
    def evaluate_dataset(
        self,
        samples: List[Dict[str, str]],
        max_new_tokens: int = 50
    ) -> List[Dict[str, str]]:
        """
        Evaluate multiple NIAH samples.
        
        Args:
            samples: List of samples to evaluate
            max_new_tokens: Max tokens per generation
        
        Returns:
            List of evaluation results
        """
        results = []
        
        for sample in samples:
            result = self.evaluate_sample(sample, max_new_tokens=max_new_tokens)
            results.append(result)
        
        return results
