"""Typed configuration dataclasses for WLASL300 Sign Language Recognition.

This module defines a hierarchy of frozen dataclasses that mirror the structure
of ``config/config.yaml``. The top-level :class:`Config` object is the single
source of truth for all hyperparameters and paths used across training,
evaluation, and inference.

Typical usage::

    from config.base_config import Config
    cfg = Config.from_yaml("config/config.yaml")
    print(cfg.training.batch_size)
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# =============================================================================
# Sub-configs
# =============================================================================


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths for data, annotations, checkpoints, and logs.

    All relative paths are resolved from the project root at load time.
    Any path can be overridden by the corresponding environment variable
    defined in ``.env``.

    Attributes:
        processed_dir: Directory for preprocessed video tensors (.pt files).
        features_dir: Directory for offline I3D feature cache vectors.
        annotations_dir: Directory containing all annotation output files.
        annotations_file: Per-video metadata JSON produced by build_annotations.py.
        vocab_file: Ordered list of 300 class names (index == class ID).
        splits_file: Train / val / test video ID assignments.
        embeddings_file: L2-normalised Word2Vec embedding matrix (300, 300).
        preprocessing_dir: Root directory of pre-extracted JPG frames.
            Layout: ``preprocessing/<split>/frames/<class_idx>/<video_id>/``.
        folder2label_file: Path to ``folder2label_str.txt`` mapping
            ``<class_idx> <label_str>`` — one line per class.
        word2vec_bin: Google News Word2Vec binary file path.
        trained_models_dir: Root directory for all checkpoints.
        best_checkpoint_dir: Saved when validation top-1 accuracy improves.
        latest_checkpoint_dir: Overwritten every epoch.
        log_dir: Root directory for logs and coverage reports.
        plots_dir: Directory for saved matplotlib / plotly figures.
    """

    processed_dir: str = "dataset/processed"
    features_dir: str = "dataset/features"
    annotations_dir: str = "dataset/annotations"
    annotations_file: str = "dataset/annotations/annotations.json"
    vocab_file: str = "dataset/annotations/vocab.json"
    splits_file: str = "dataset/annotations/splits.json"
    embeddings_file: str = "dataset/annotations/word2vec_embeddings.npy"
    preprocessing_dir: str = "dataset/raw/preprocessing"
    folder2label_file: str = "dataset/raw/folder2label_str.txt"
    word2vec_bin: str = "trained_models/embeddings/GoogleNews-vectors-negative300.bin"
    trained_models_dir: str = "trained_models"
    best_checkpoint_dir: str = "trained_models/best"
    latest_checkpoint_dir: str = "trained_models/latest"
    log_dir: str = "logs"
    plots_dir: str = "logs/plots"

    def __post_init__(self) -> None:
        """Apply environment variable overrides after dataclass initialisation."""
        env_overrides = {
            "word2vec_bin": "WORD2VEC_BIN_PATH",
        }
        for attr, env_var in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value:
                # frozen=True prevents direct assignment; use object.__setattr__
                object.__setattr__(self, attr, env_value)

    def make_dirs(self) -> None:
        """Create all output directories if they do not already exist.

        Safe to call multiple times — uses ``exist_ok=True`` internally.
        Raises:
            OSError: If a directory cannot be created due to permission errors.
        """
        dirs = [
            self.processed_dir,
            self.features_dir,
            self.annotations_dir,
            self.best_checkpoint_dir,
            self.latest_checkpoint_dir,
            self.log_dir,
            self.plots_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetConfig:
    """Video loading and preprocessing settings.

    Attributes:
        num_classes: Total number of sign language classes (300 for WLASL300).
        num_frames: Number of frames per clip. Pre-extracted clips contain
            exactly 16 frames. For live-feed inference from raw video, frames
            are sampled to match this count.
        frame_size: Spatial resolution (H = W) of each frame. Pre-extracted
            frames are 256x256. Used as the crop size for raw video inputs.
        resize_size: Shorter-side resize target before cropping. Set equal to
            ``frame_size`` for pre-extracted frames (no resize needed).
        mean: Per-channel ImageNet mean for normalisation (RGB order).
        std: Per-channel ImageNet std for normalisation (RGB order).
        train_split: Fraction of data assigned to training.
        val_split: Fraction of data assigned to validation.
        test_split: Fraction of data assigned to test.
        split_seed: Random seed for reproducible stratified splitting.
        num_workers: DataLoader worker processes (0 = single-process).
        pin_memory: Pin CPU tensors for faster GPU transfer.
        loop_short_videos: Loop clips shorter than ``num_frames`` to fill them.
        max_duration_seconds: Clips longer than this are trimmed before sampling.
        class_balanced_sampling: Use ``WeightedRandomSampler`` on the training
            loader so every class contributes equally regardless of clip count.
            Rare classes (14 clips) are upsampled to match common ones (40 clips).
            Has no effect on val or test loaders.
    """

    num_classes: int = 300
    num_frames: int = 16
    frame_size: int = 256
    resize_size: int = 256
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    split_seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    loop_short_videos: bool = True
    max_duration_seconds: float = 10.0
    class_balanced_sampling: bool = True

    def __post_init__(self) -> None:
        """Validate split fractions sum to 1.0.

        Raises:
            ValueError: If train + val + test splits do not sum to 1.0
                (within floating-point tolerance).
        """
        total = self.train_split + self.val_split + self.test_split
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"train_split + val_split + test_split must equal 1.0, got {total:.4f}"
            )


@dataclass(frozen=True)
class AugmentationConfig:
    """Video augmentation settings applied during training only.

    All augmentations are disabled at validation and test time — only
    deterministic centre crop and normalisation are applied there.

    Attributes:
        random_horizontal_flip: Randomly mirror frames left-right.
        horizontal_flip_prob: Probability of applying the horizontal flip.
        random_crop: Use random crop during training (centre crop at val/test).
        random_crop_size: Spatial size of the random crop output.
        color_jitter: Randomly perturb brightness, contrast, saturation, hue.
        color_jitter_brightness: Max brightness perturbation magnitude.
        color_jitter_contrast: Max contrast perturbation magnitude.
        color_jitter_saturation: Max saturation perturbation magnitude.
        color_jitter_hue: Max hue perturbation magnitude (range: 0.0–0.5).
        temporal_jitter: Randomly shift the frame sampling grid.
        temporal_jitter_frames: Maximum shift in frames (±N).
        speed_perturbation: Resample video at a random playback rate.
        speed_min: Minimum playback rate multiplier.
        speed_max: Maximum playback rate multiplier.
        random_noise: Add Gaussian noise to normalised frame tensors.
        noise_std: Standard deviation of the Gaussian noise.
    """

    random_horizontal_flip: bool = True
    horizontal_flip_prob: float = 0.5
    random_crop: bool = True
    random_crop_size: int = 224
    color_jitter: bool = True
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.05
    temporal_jitter: bool = True
    temporal_jitter_frames: int = 4
    speed_perturbation: bool = True
    speed_min: float = 0.75
    speed_max: float = 1.25
    random_noise: bool = False
    noise_std: float = 0.01


@dataclass(frozen=True)
class ModelConfig:
    """Neural network architecture settings.

    Attributes:
        backbone: I3D backbone variant name for pytorchvideo / torchvision.
        pretrained: Load Kinetics-400 pretrained weights into the backbone.
        backbone_output_dim: Feature dimension after global average pooling.
        embedding_dim: Output dimension of the projection head (= Word2Vec dim).
        projection_hidden_dim: Hidden layer width inside the projection head.
        dropout: Dropout probability applied inside the projection head.
        l2_normalize: Apply L2 normalisation to the final embedding vector.
    """

    backbone: str = "i3d_r50"
    pretrained: bool = True
    backbone_output_dim: int = 512
    embedding_dim: int = 300
    projection_hidden_dim: int = 512
    dropout: float = 0.4
    l2_normalize: bool = True


@dataclass(frozen=True)
class PhaseConfig:
    """Settings for a single training phase.

    Attributes:
        epochs: Number of epochs to run this phase.
        learning_rate: Initial learning rate for this phase.
        freeze_backbone: Whether to freeze all backbone parameters.
        unfreeze_last_n_blocks: Number of backbone blocks to unfreeze from the
            end. Ignored when ``freeze_backbone`` is True. ``None`` means
            unfreeze the entire backbone.
    """

    epochs: int = 10
    learning_rate: float = 1e-3
    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """Full training loop configuration across all phases.

    Attributes:
        epochs: Total number of training epochs.
        batch_size: Video clips per GPU per optimiser step.
        grad_accumulation_steps: Gradient accumulation steps before optimizer.step().
        mixed_precision: Enable torch.cuda.amp automatic mixed precision.
        grad_clip_norm: Global gradient norm clipping value. ``None`` disables.
        seed: Global random seed for PyTorch, NumPy, and Python's random module.
        loss: Primary loss function identifier (currently only ``"cosine"``).
        triplet_loss_weight: Weight of the auxiliary triplet loss (0.0 = disabled).
        triplet_margin: Margin for the triplet loss function.
        phase1: Warm-up phase — projection head only, backbone frozen.
        phase2: Fine-tune phase — last N backbone blocks + projection head.
        phase3: Full fine-tune phase — all layers trainable.
    """

    epochs: int = 60
    batch_size: int = 8
    grad_accumulation_steps: int = 2
    mixed_precision: bool = True
    grad_clip_norm: float | None = 1.0
    seed: int = 42
    loss: str = "cosine"
    triplet_loss_weight: float = 0.3
    triplet_margin: float = 0.2
    phase1: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(epochs=10, learning_rate=1e-3, freeze_backbone=True)
    )
    phase2: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(
            epochs=30, learning_rate=1e-4, freeze_backbone=False, unfreeze_last_n_blocks=2
        )
    )
    phase3: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(
            epochs=20, learning_rate=1e-5, freeze_backbone=False, unfreeze_last_n_blocks=None
        )
    )


@dataclass(frozen=True)
class OptimiserConfig:
    """Optimiser hyperparameters.

    Attributes:
        name: Optimiser name. Supported: ``"adamw"``, ``"adam"``, ``"sgd"``.
        weight_decay: L2 regularisation coefficient.
        betas: Exponential decay rates for AdamW gradient moment estimates.
        eps: Numerical stability term for AdamW.
    """

    name: str = "adamw"
    weight_decay: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning rate scheduler settings.

    Attributes:
        name: Scheduler identifier. Supported: ``"cosine_warm_restarts"``,
            ``"step"``, ``"none"``.
        T_0: Epochs per restart cycle for cosine warm restarts.
        T_mult: Cycle length multiplier applied after each restart.
        min_lr: Minimum LR at the trough of each cosine cycle.
        warmup_steps: Number of linear warm-up steps at the start of each phase.
    """

    name: str = "cosine_warm_restarts"
    T_0: int = 10
    T_mult: int = 2
    min_lr: float = 1e-6
    warmup_steps: int = 100


@dataclass(frozen=True)
class TemporalNeckConfig:
    """Hyperparameters for the temporal transformer neck.

    Attributes:
        d_model: Transformer internal dimension.  All attention and FFN
            operations run at this width.  Must be divisible by ``nhead``.
        nhead: Number of self-attention heads.
        num_layers: Number of stacked transformer encoder layers.
        dim_feedforward: FFN hidden dimension inside each encoder layer.
        dropout: Dropout probability inside the transformer and on positional
            encoding.
    """

    d_model: int = 256
    nhead: int = 8
    num_layers: int = 2
    dim_feedforward: int = 1024
    dropout: float = 0.1


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early stopping criteria.

    Attributes:
        enabled: Whether early stopping is active.  Set to ``False`` to
            disable it entirely and always train for the full epoch count.
        monitor: Name of the metric to watch (e.g. ``"val_top1_accuracy"``).
        patience: Epochs without improvement before training is halted.
            Only checked when ``enabled`` is ``True``.
        min_delta: Minimum absolute change to count as an improvement.
        mode: ``"max"`` if higher is better, ``"min"`` if lower is better.
    """

    enabled: bool = True
    monitor: str = "val_top1_accuracy"
    patience: int = 5
    min_delta: float = 0.001
    mode: str = "max"


@dataclass(frozen=True)
class CheckpointingConfig:
    """Checkpoint saving policy.

    Attributes:
        save_every_epoch: Save a checkpoint after every epoch to ``latest/``.
        save_best: Save to ``best/`` whenever the best metric improves.
        best_metric: Metric name used to determine the best checkpoint.
        keep_last_n: Maximum recent checkpoints to retain in ``latest/``.
            ``None`` retains all.
    """

    save_every_epoch: bool = True
    save_best: bool = True
    best_metric: str = "val_top1_accuracy"
    keep_last_n: int | None = 3


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation and retrieval settings.

    Attributes:
        topk: List of k values for top-k accuracy computation.
        run_test_eval: Run full test-set evaluation after training completes.
        retrieval_k: Number of nearest neighbours returned per query.
    """

    topk: list[int] = field(default_factory=lambda: [1, 5])
    run_test_eval: bool = True
    retrieval_k: int = 10


@dataclass(frozen=True)
class LoggingConfig:
    """Experiment tracking and logging settings.

    Attributes:
        logger: Experiment tracker to use. One of ``"wandb"``, ``"mlflow"``,
            ``"none"``.
        log_every_n_steps: Log metrics every N training steps.
        log_embedding_samples: Log a sample of predicted embeddings each epoch.
        embedding_sample_size: Number of validation samples to visualise.
        log_level: Loguru log level string (``"DEBUG"``, ``"INFO"``, etc.).
    """

    logger: str = "none"
    log_every_n_steps: int = 10
    log_embedding_samples: bool = True
    embedding_sample_size: int = 50
    log_level: str = "INFO"


@dataclass(frozen=True)
class InferenceConfig:
    """Inference script settings.

    Attributes:
        default_checkpoint: Path to the checkpoint loaded by inference.py.
        top_k: Number of top predictions returned per video.
        device: Compute device — ``"cuda"``, ``"cpu"``, or ``"auto"``.
        batch_size: Batch size for directory-level batch inference.
    """

    default_checkpoint: str = "trained_models/best/checkpoint.pt"
    top_k: int = 5
    device: str = "auto"
    batch_size: int = 16


# =============================================================================
# Top-level config
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Top-level configuration object for the WLASL300 project.

    Aggregates all sub-configs into a single object that is passed around
    the codebase. Constructed via :meth:`from_yaml` rather than directly.

    Attributes:
        paths: Filesystem path settings.
        dataset: Video loading and preprocessing settings.
        augmentation: Training-time augmentation settings.
        model: Neural network architecture settings.
        training: Training loop and phased fine-tuning settings.
        optimiser: Optimiser hyperparameters.
        scheduler: Learning rate scheduler settings.
        early_stopping: Early stopping criteria.
        checkpointing: Checkpoint saving policy.
        evaluation: Evaluation and retrieval settings.
        logging: Experiment tracking and logging settings.
        inference: Inference script settings.

    Example::

        cfg = Config.from_yaml("config/config.yaml")
        cfg.paths.make_dirs()
        print(cfg.training.batch_size)   # 8
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimiser: OptimiserConfig = field(default_factory=OptimiserConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    temporal_neck: TemporalNeckConfig = field(default_factory=TemporalNeckConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Config:
        """Load and validate a Config from a YAML file.

        Reads the YAML file at ``yaml_path``, maps its nested structure onto
        the corresponding dataclass hierarchy, and returns a fully populated
        frozen :class:`Config` instance.

        Args:
            yaml_path: Path to the YAML configuration file. Typically
                ``"config/config.yaml"``.

        Returns:
            A fully populated, frozen :class:`Config` instance.

        Raises:
            FileNotFoundError: If ``yaml_path`` does not exist.
            yaml.YAMLError: If the YAML file is malformed.
            ValueError: If any sub-config validation fails (e.g. split fractions
                do not sum to 1.0).

        Example::

            cfg = Config.from_yaml("config/config.yaml")
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with yaml_path.open("r") as f:
            raw: dict = yaml.safe_load(f)

        def _get(section: str, defaults_cls: type) -> dict:
            """Extract a YAML section, falling back to dataclass defaults."""
            return raw.get(section, {})

        return cls(
            paths=_build(PathsConfig, _get("paths", PathsConfig)),
            dataset=_build(DatasetConfig, _get("dataset", DatasetConfig)),
            augmentation=_build(AugmentationConfig, _get("augmentation", AugmentationConfig)),
            model=_build(ModelConfig, _get("model", ModelConfig)),
            training=_build_training(_get("training", TrainingConfig)),
            optimiser=_build(OptimiserConfig, _get("optimiser", OptimiserConfig)),
            scheduler=_build(SchedulerConfig, _get("scheduler", SchedulerConfig)),
            early_stopping=_build(EarlyStoppingConfig, _get("early_stopping", EarlyStoppingConfig)),
            temporal_neck=_build(TemporalNeckConfig, _get("temporal_neck", TemporalNeckConfig)),
            checkpointing=_build(CheckpointingConfig, _get("checkpointing", CheckpointingConfig)),
            evaluation=_build(EvaluationConfig, _get("evaluation", EvaluationConfig)),
            logging=_build(LoggingConfig, _get("logging", LoggingConfig)),
            inference=_build(InferenceConfig, _get("inference", InferenceConfig)),
        )

    def to_dict(self) -> dict:
        """Serialise the config to a plain nested dictionary.

        Useful for logging the full config to W&B, MLflow, or a JSON file.

        Returns:
            A nested ``dict`` representation of the entire config tree.
        """
        return dataclasses.asdict(self)


# =============================================================================
# Private helpers
# =============================================================================


def _build(cls: type, data: dict) -> object:
    """Instantiate a frozen dataclass from a dict, ignoring unknown keys.

    Args:
        cls: The dataclass type to instantiate.
        data: Dictionary of field values (may contain extra keys from YAML).

    Returns:
        An instance of ``cls`` populated with values from ``data``.
    """
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


def _build_training(data: dict) -> TrainingConfig:
    """Build a :class:`TrainingConfig` including its nested phase configs.

    Handles the ``phase1``, ``phase2``, ``phase3`` sub-dicts inside the
    ``training`` YAML section.

    Args:
        data: The ``training`` section dict from the YAML file.

    Returns:
        A fully populated :class:`TrainingConfig` instance.
    """
    phase1_data = data.pop("phase1", {})
    phase2_data = data.pop("phase2", {})
    phase3_data = data.pop("phase3", {})

    valid_fields = {f.name for f in dataclasses.fields(TrainingConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    return TrainingConfig(
        **filtered,
        phase1=_build(PhaseConfig, phase1_data),
        phase2=_build(PhaseConfig, phase2_data),
        phase3=_build(PhaseConfig, phase3_data),
    )
