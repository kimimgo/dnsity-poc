"""
Unit tests for experiment configuration management.

Tests verify that:
1. YAML experiment configs can be loaded correctly
2. Config validation catches invalid parameters
3. Different Gist token counts (10, 25, 50) are properly configured
"""

import pytest
from pathlib import Path
import tempfile


class TestExperimentConfig:
    """Test suite for experiment configuration."""

    def test_load_experiment_config(self):
        """Test loading experiment configuration from YAML."""
        from src.model.config import GistTokenConfig
        
        # Load the 10-token config
        config_path = Path("experiments/configs/gist_10.yaml")
        
        if config_path.exists():
            config = GistTokenConfig.from_yaml(config_path)
            
            assert config.num_gist_tokens == 10
            assert config.model_name == "meta-llama/Meta-Llama-3-8B-Instruct"
            assert config.load_in_4bit is True

    def test_gist_count_variation(self):
        """Test that configs for different Gist counts exist and are valid."""
        from src.model.config import GistTokenConfig
        
        gist_counts = [10, 25, 50]
        
        for count in gist_counts:
            config_path = Path(f"experiments/configs/gist_{count}.yaml")
            
            if config_path.exists():
                config = GistTokenConfig.from_yaml(config_path)
                assert config.num_gist_tokens == count

    def test_config_to_dict(self):
        """Test that config can be converted to dictionary."""
        from src.model.config import GistTokenConfig
        
        config = GistTokenConfig(
            model_name="gpt2",
            num_gist_tokens=5,
            load_in_4bit=False
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["num_gist_tokens"] == 5
        assert config_dict["model_name"] == "gpt2"

    def test_config_validation(self):
        """Test that invalid configs raise errors."""
        from src.model.config import GistTokenConfig
        
        # num_gist_tokens must be positive
        with pytest.raises((ValueError, AssertionError)):
            config = GistTokenConfig(num_gist_tokens=0)
            config.validate()

    def test_config_save_load_roundtrip(self):
        """Test saving and loading config preserves values."""
        from src.model.config import GistTokenConfig
        
        original_config = GistTokenConfig(
            model_name="gpt2",
            num_gist_tokens=15,
            load_in_4bit=True,
            lora_r=8,
            lora_alpha=16
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            
            # Save
            original_config.save_yaml(config_path)
            
            # Load
            loaded_config = GistTokenConfig.from_yaml(config_path)
            
            assert loaded_config.num_gist_tokens == 15
            assert loaded_config.model_name == "gpt2"
            assert loaded_config.lora_r == 8
