#!/usr/bin/env python3
"""
Configuration management for ST-Mirror.

Uses YAML for human-readable defaults with Pydantic for validation.
Environment variables can override any setting.
"""

import os
from pathlib import Path
from typing import Optional, Dict
from pydantic import BaseModel, Field
import yaml


class ModelConfig(BaseModel):
    """Configuration for a model."""
    name: str
    temperature: float = 0.3
    max_tokens: int
    reasoning_effort: Optional[str] = None
    retries: int = 3
    timeout_seconds: int = 60


class Phase1Config(BaseModel):
    """Phase 1 evidence extraction config."""
    primary: ModelConfig
    fallback_high_reasoning: ModelConfig
    fallback_kimi: ModelConfig


class Phase2Config(BaseModel):
    """Phase 2 synthesis config."""
    name: str
    temperature: float
    max_tokens: int
    retries: int
    timeout_seconds: int


class AggregationConfig(BaseModel):
    """Aggregation model config."""
    name: str
    temperature: float
    max_tokens: int
    retries: int
    timeout_week_seconds: int
    timeout_month_seconds: int
    timeout_year_seconds: int
    timeout_life_seconds: int


class ClassificationConfig(BaseModel):
    """Classification model config."""
    name: str
    retries: int
    timeout_seconds: int


class UserConfig(BaseModel):
    """User profile settings."""
    name: str = "User"
    pronouns: str = "they/them"


class ModelsConfig(BaseModel):
    """All model configurations."""
    phase1_evidence: Phase1Config
    phase2_synthesis: Phase2Config
    aggregation: AggregationConfig
    classification: ClassificationConfig


class ProfilerConfig(BaseModel):
    """Profiler processing settings."""
    max_parallel: int = 5
    chunk_window_size: int = 50
    min_arc_length: int = 100


class AggregatorConfig(BaseModel):
    """Hierarchical aggregator settings."""
    max_parallel: int = 5
    max_profiles_per_aggregation: int = 15
    token_budget_threshold: int = 30000


class SampleStrategyConfig(BaseModel):
    """Classifier sampling strategy."""
    beginning: int = 15
    middle: int = 10
    end: int = 10


class ClassifierConfig(BaseModel):
    """Branch classifier settings."""
    max_parallel: int = 10
    # Heuristic prefilter thresholds
    min_messages: int = 100
    min_valid_message_ratio: float = 0.5
    repetition_check_threshold: int = 10
    # LLM classification sampling
    sample_size: int = 50
    message_char_limit: int = 200
    sample_strategy: SampleStrategyConfig


class ProcessingConfig(BaseModel):
    """Processing configurations."""
    profiler: ProfilerConfig
    aggregator: AggregatorConfig
    classifier: ClassifierConfig


class NetworkConfig(BaseModel):
    """Network settings."""
    backoff_base: int = 2
    max_connections_per_host: int = 10


class AppConfig(BaseModel):
    """Main application configuration."""
    user: UserConfig
    models: ModelsConfig
    processing: ProcessingConfig
    network: NetworkConfig
    costs: Optional[Dict[str, float]] = None

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AppConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml (defaults to ./config.yaml)

        Returns:
            Validated AppConfig instance
        """
        if config_path is None:
            # Default to config.yaml in project root
            config_path = Path(__file__).parent / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Please create config.yaml in the project root."
            )

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Apply environment variable overrides
        # Format: RP_MODELS_PHASE2_MAX_TOKENS=20000
        data = cls._apply_env_overrides(data)

        return cls(**data)

    @staticmethod
    def _apply_env_overrides(data: dict) -> dict:
        """
        Apply environment variable overrides.

        Environment variable format: RP_SECTION_SUBSECTION_KEY
        Examples:
            RP_MODELS_PHASE2_SYNTHESIS_MAX_TOKENS=20000
            RP_PROCESSING_AGGREGATOR_MAX_PARALLEL=5
        """
        for key, value in os.environ.items():
            if not key.startswith("RP_"):
                continue

            # Parse key path: RP_MODELS_PHASE2_SYNTHESIS_MAX_TOKENS
            parts = key[3:].lower().split("_")  # Remove RP_ prefix

            if len(parts) < 2:
                continue

            # Navigate nested dict
            current = data
            for part in parts[:-1]:
                if part not in current:
                    break
                current = current[part]
            else:
                # Set the value (with type conversion)
                final_key = parts[-1]
                if final_key in current:
                    # Try to preserve type from config
                    original_value = current[final_key]
                    if isinstance(original_value, bool):
                        current[final_key] = value.lower() in ('true', '1', 'yes')
                    elif isinstance(original_value, int):
                        current[final_key] = int(value)
                    elif isinstance(original_value, float):
                        current[final_key] = float(value)
                    else:
                        current[final_key] = value

        return data


# Singleton instance - load once at module import
try:
    config = AppConfig.load()
except FileNotFoundError as e:
    # Provide helpful error message
    print(f"ERROR: {e}")
    print("Run the toolkit from the project root directory.")
    raise
