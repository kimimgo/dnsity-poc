"""
Configuration management for Gist Token models.

Provides utilities to load model configurations from YAML files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class GistTokenConfig:
    """Configuration for Gist Token model."""

    # Model settings
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    num_gist_tokens: int = 10
    load_in_4bit: bool = True

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = None
    modules_to_save: list[str] = None

    # Training settings
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    max_steps: int = 1000
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if self.modules_to_save is None:
            self.modules_to_save = ["embed_tokens", "lm_head"]

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "GistTokenConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            GistTokenConfig instance
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Flatten nested structure (e.g., {model: {name: ...}} -> {model_name: ...})
        flat_config = {}

        if "model" in config_dict:
            for key, value in config_dict["model"].items():
                flat_config[f"model_{key}"] = value if key != "name" else value
                if key == "name":
                    flat_config["model_name"] = value

        if "lora" in config_dict:
            for key, value in config_dict["lora"].items():
                flat_config[f"lora_{key}"] = value

        if "training" in config_dict:
            for key, value in config_dict["training"].items():
                flat_config[key] = value

        return cls(**flat_config)

    def to_yaml(self, config_path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            config_path: Path to save YAML config
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Structure as nested dict
        config_dict = {
            "model": {
                "name": self.model_name,
                "num_gist_tokens": self.num_gist_tokens,
                "load_in_4bit": self.load_in_4bit,
            },
            "lora": {
                "r": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "target_modules": self.lora_target_modules,
                "modules_to_save": self.modules_to_save,
            },
            "training": {
                "learning_rate": self.learning_rate,
                "warmup_steps": self.warmup_steps,
                "max_steps": self.max_steps,
                "per_device_train_batch_size": self.per_device_train_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
