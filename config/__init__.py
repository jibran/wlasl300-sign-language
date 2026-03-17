"""Configuration package for WLASL300 Sign Language Recognition.

Exposes the top-level :class:`~config.base_config.Config` dataclass and its
convenience loader so the rest of the codebase can import cleanly::

    from config import Config
    cfg = Config.from_yaml("config/config.yaml")
"""

from config.base_config import (
    AugmentationConfig,
    CheckpointingConfig,
    Config,
    DatasetConfig,
    EarlyStoppingConfig,
    EvaluationConfig,
    InferenceConfig,
    LoggingConfig,
    ModelConfig,
    OptimiserConfig,
    PathsConfig,
    PhaseConfig,
    SchedulerConfig,
    TrainingConfig,
)

__all__ = [
    "Config",
    "PathsConfig",
    "DatasetConfig",
    "AugmentationConfig",
    "ModelConfig",
    "PhaseConfig",
    "TrainingConfig",
    "OptimiserConfig",
    "SchedulerConfig",
    "EarlyStoppingConfig",
    "CheckpointingConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "InferenceConfig",
]
