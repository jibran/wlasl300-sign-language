"""Main training script for WLASL300 sign language recognition.

Runs the full three-phase training pipeline:

- **Phase 1** (epochs 1–10): Backbone frozen, projection head only.
- **Phase 2** (epochs 11–40): Last N backbone blocks unfrozen.
- **Phase 3** (epochs 41–60): Full backbone fine-tuning at low LR.

Each phase transition automatically adjusts the optimiser's parameter groups
and learning rate.  Training can be resumed from any checkpoint.

Usage::

    # Standard run
    uv run python train/train.py --config config/config.yaml

    # Override hyperparameters
    uv run python train/train.py \\
        --config config/config.yaml \\
        --training.batch_size 16 \\
        --training.phase1.learning_rate 5e-4

    # Resume from latest checkpoint
    uv run python train/train.py \\
        --config config/config.yaml \\
        --resume trained_models/latest/checkpoint.pt

    # Choose experiment tracker
    uv run python train/train.py \\
        --config config/config.yaml \\
        --logger wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

# Project imports
from config.base_config import Config
from dataset.data.wlasl_dataset import build_dataloaders
from models.sign_model import SignModel
from utils.metrics import EpochMetrics, MetricTracker, throughput
from utils.visualization import plot_training_summary

log = logging.getLogger(__name__)


# =============================================================================
# Seed and reproducibility
# =============================================================================


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducible training runs.

    Sets seeds for Python's ``random``, NumPy, PyTorch CPU and CUDA,
    and configures cuDNN for deterministic operation.

    Args:
        seed: Integer random seed.  Use the same value across runs to
            reproduce results exactly.

    Note:
        ``torch.backends.cudnn.deterministic = True`` may reduce GPU
        throughput.  Set it to ``False`` for maximum speed if exact
        reproducibility is not required.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info("Random seed set to %d", seed)


# =============================================================================
# Device selection
# =============================================================================


def get_device() -> torch.device:
    """Select the best available compute device.

    Returns:
        ``torch.device("cuda")`` if a CUDA GPU is available,
        ``torch.device("mps")`` for Apple Silicon,
        otherwise ``torch.device("cpu")``.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(
            "Using CUDA device: %s (%.1f GB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        log.warning("No GPU found — training on CPU will be very slow")
    return device


# =============================================================================
# Optimiser and scheduler construction
# =============================================================================


def build_optimiser(model: SignModel, cfg: Config, phase: int) -> AdamW:
    """Construct an AdamW optimiser with phase-appropriate parameter groups.

    Uses separate parameter groups for backbone and head so that different
    learning rates can be applied to each.  Frozen parameters are excluded
    from all groups.

    Args:
        model: The :class:`~models.sign_model.SignModel` to optimise.
        cfg: Full project config.
        phase: Training phase (1, 2, or 3).

    Returns:
        Configured :class:`torch.optim.AdamW` instance.
    """
    opt_cfg = cfg.optimiser

    # Determine per-phase learning rates
    phase_lr_map = {
        1: cfg.training.phase1.learning_rate,
        2: cfg.training.phase2.learning_rate,
        3: cfg.training.phase3.learning_rate,
    }
    phase_lr = phase_lr_map[phase]

    # Collect trainable parameters, separating backbone from head
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]

    param_groups = []
    if backbone_params:
        # Backbone uses a lower LR than the head to protect pretrained features
        param_groups.append(
            {
                "params": backbone_params,
                "lr": phase_lr * 0.1,  # 10× lower than head during partial unfreeze
                "name": "backbone",
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": phase_lr,
                "name": "head",
            }
        )

    if not param_groups:
        raise ValueError(
            "No trainable parameters found. "
            "Check that apply_phase() was called before build_optimiser()."
        )

    optimiser = AdamW(
        param_groups,
        weight_decay=opt_cfg.weight_decay,
        betas=tuple(opt_cfg.betas),
        eps=opt_cfg.eps,
    )

    trainable_count = sum(p.numel() for g in param_groups for p in g["params"])
    log.info(
        "Optimiser: AdamW  phase=%d  lr=%.2e  trainable_params=%s",
        phase,
        phase_lr,
        f"{trainable_count:,}",
    )
    return optimiser


def build_scheduler(
    optimiser: AdamW,
    cfg: Config,
) -> CosineAnnealingWarmRestarts:
    """Construct a cosine annealing LR scheduler with warm restarts.

    Args:
        optimiser: The AdamW optimiser to schedule.
        cfg: Full project config.  Reads ``cfg.scheduler``.

    Returns:
        Configured :class:`~torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.
    """
    sched_cfg = cfg.scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimiser,
        T_0=sched_cfg.T_0,
        T_mult=sched_cfg.T_mult,
        eta_min=sched_cfg.min_lr,
    )
    return scheduler


# =============================================================================
# Warmup helper
# =============================================================================


def linear_warmup(
    optimiser: AdamW,
    step: int,
    warmup_steps: int,
    base_lrs: list[float],
) -> None:
    """Apply linear learning-rate warmup for the first ``warmup_steps`` steps.

    Scales each parameter group's LR linearly from 0 to its base value.
    Called once per training step until ``step >= warmup_steps``.

    Args:
        optimiser: The AdamW optimiser whose LR groups to update.
        step: Current global training step (0-indexed).
        warmup_steps: Number of steps over which to warm up.
        base_lrs: List of target LR values, one per parameter group.
    """
    if warmup_steps <= 0 or step >= warmup_steps:
        return
    scale = (step + 1) / warmup_steps
    for group, base_lr in zip(optimiser.param_groups, base_lrs, strict=True):
        group["lr"] = base_lr * scale


# =============================================================================
# Single-epoch train and eval loops
# =============================================================================


def train_one_epoch(
    model: SignModel,
    loader: DataLoader,
    optimiser: AdamW,
    scheduler: CosineAnnealingWarmRestarts,
    scaler: GradScaler,
    class_embeddings: torch.Tensor,
    cfg: Config,
    epoch: int,
    global_step: int,
    warmup_steps: int,
    base_lrs: list[float],
    logger: object | None,
    device: torch.device,
) -> tuple[EpochMetrics, int]:
    """Run one full training epoch.

    Iterates over all batches in ``loader``, computes the combined
    cosine + triplet loss, performs a backward pass with optional mixed
    precision and gradient accumulation, and updates the LR scheduler.

    Args:
        model: The :class:`~models.sign_model.SignModel` in training mode.
        loader: Training :class:`~torch.utils.data.DataLoader`.
        optimiser: AdamW optimiser.
        scheduler: Cosine warm-restart LR scheduler.
        scaler: AMP :class:`~torch.amp.GradScaler`.
        class_embeddings: Class embedding matrix ``(C, D)`` on ``device``.
        cfg: Full project config.
        epoch: Current epoch index (0-based).
        global_step: Global step counter carried across epochs.
        warmup_steps: Number of steps for LR warm-up.
        base_lrs: Target LR per parameter group (used for warmup scaling).
        logger: Experiment logger instance (W&B / MLflow / ``None``).
        device: Compute device.

    Returns:
        A tuple of:
            - :class:`~utils.metrics.EpochMetrics` for this epoch.
            - Updated ``global_step`` counter.
    """
    model.train()
    model.backbone.set_train_mode(is_train=True)

    tracker = MetricTracker(
        num_classes=cfg.dataset.num_classes,
        topk=tuple(cfg.evaluation.topk),
    )
    accum_steps = cfg.training.grad_accumulation_steps
    log_every = cfg.logging.log_every_n_steps
    trip_weight = cfg.training.triplet_loss_weight
    trip_margin = cfg.training.triplet_margin
    use_amp = cfg.training.mixed_precision and device.type == "cuda"

    t0 = time.perf_counter()
    optimiser.zero_grad()

    for batch_idx, (videos, embeddings, labels) in enumerate(loader):
        videos = videos.to(device, non_blocking=True)  # (B, 3, T, H, W)
        embeddings = embeddings.to(device, non_blocking=True)  # (B, D)
        labels = labels.to(device, non_blocking=True)  # (B,)

        # ------------------------------------------------------------------ #
        # Linear LR warmup
        # ------------------------------------------------------------------ #
        linear_warmup(optimiser, global_step, warmup_steps, base_lrs)

        # ------------------------------------------------------------------ #
        # Forward pass (mixed precision optional)
        # ------------------------------------------------------------------ #
        with autocast(device_type=device.type, enabled=use_amp):
            pred_emb = model(videos)  # (B, D)
            total_loss, loss_dict = model.combined_loss(
                pred=pred_emb,
                target=embeddings,
                triplet_weight=trip_weight,
                triplet_margin=trip_margin,
            )
            # Scale loss for gradient accumulation
            scaled_loss = total_loss / accum_steps

        # ------------------------------------------------------------------ #
        # Backward pass
        # ------------------------------------------------------------------ #
        scaler.scale(scaled_loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            # Gradient clipping before unscaling
            if cfg.training.grad_clip_norm:
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()
            scheduler.step(epoch + batch_idx / len(loader))
            global_step += 1

        # ------------------------------------------------------------------ #
        # Accumulate metrics
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            tracker.update(
                pred_embeddings=pred_emb,
                class_embeddings=class_embeddings,
                true_label_indices=labels,
                loss_dict=loss_dict,
            )

        # ------------------------------------------------------------------ #
        # Step-level logging
        # ------------------------------------------------------------------ #
        if logger is not None and global_step % log_every == 0:
            current_lr = optimiser.param_groups[-1]["lr"]
            _log_step(logger, loss_dict, current_lr, global_step)

    # Compute epoch metrics
    elapsed = time.perf_counter() - t0
    results = tracker.compute()
    n_samples = tracker.num_samples

    metrics = EpochMetrics(
        split="train",
        epoch=epoch,
        loss=results.get("loss", 0.0),
        cosine_loss=results.get("cosine_loss", 0.0),
        triplet_loss=results.get("triplet_loss", 0.0),
        top1=results.get("top1", 0.0),
        top5=results.get("top5", 0.0),
        mean_cosine_sim=results.get("mean_cosine_sim", 0.0),
        num_samples=n_samples,
        throughput_clips_per_sec=throughput(n_samples, elapsed),
    )
    return metrics, global_step


@torch.no_grad()
def evaluate(
    model: SignModel,
    loader: DataLoader,
    class_embeddings: torch.Tensor,
    cfg: Config,
    epoch: int,
    split: str,
    device: torch.device,
    compute_per_class: bool = False,
    vocab: list[str] | None = None,
) -> EpochMetrics:
    """Run evaluation on a val or test split.

    Args:
        model: The :class:`~models.sign_model.SignModel` in eval mode.
        loader: Val or test :class:`~torch.utils.data.DataLoader`.
        class_embeddings: Class embedding matrix ``(C, D)`` on ``device``.
        cfg: Full project config.
        epoch: Current epoch index (0-based).
        split: Split name — ``"val"`` or ``"test"``.
        device: Compute device.
        compute_per_class: If ``True``, compute per-class top-1 accuracy.
        vocab: Class name list for per-class labelling.

    Returns:
        :class:`~utils.metrics.EpochMetrics` for this split.
    """
    model.eval()

    tracker = MetricTracker(
        num_classes=cfg.dataset.num_classes,
        topk=tuple(cfg.evaluation.topk),
    )
    use_amp = cfg.training.mixed_precision and device.type == "cuda"

    t0 = time.perf_counter()
    for videos, embeddings, labels in loader:
        videos = videos.to(device, non_blocking=True)
        embeddings = embeddings.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            pred_emb = model(videos)
            _, loss_dict = model.combined_loss(
                pred=pred_emb,
                target=embeddings,
                triplet_weight=cfg.training.triplet_loss_weight,
                triplet_margin=cfg.training.triplet_margin,
            )

        tracker.update(
            pred_embeddings=pred_emb,
            class_embeddings=class_embeddings,
            true_label_indices=labels,
            loss_dict=loss_dict,
        )

    elapsed = time.perf_counter() - t0
    results = tracker.compute(compute_per_class=compute_per_class, vocab=vocab)
    n_samples = tracker.num_samples

    per_class = {
        k.replace("per_class/", ""): v for k, v in results.items() if k.startswith("per_class/")
    }

    metrics = EpochMetrics(
        split=split,
        epoch=epoch,
        loss=results.get("loss", 0.0),
        cosine_loss=results.get("cosine_loss", 0.0),
        triplet_loss=results.get("triplet_loss", 0.0),
        top1=results.get("top1", 0.0),
        top5=results.get("top5", 0.0),
        mean_cosine_sim=results.get("mean_cosine_sim", 0.0),
        num_samples=n_samples,
        throughput_clips_per_sec=throughput(n_samples, elapsed),
        per_class_top1=per_class,
    )
    return metrics


# =============================================================================
# Checkpoint management
# =============================================================================


def _save_latest(
    model: SignModel,
    optimiser: AdamW,
    scheduler: CosineAnnealingWarmRestarts,
    epoch: int,
    metrics: EpochMetrics,
    cfg: Config,
    keep_last_n: int | None,
) -> None:
    """Save a checkpoint to ``trained_models/latest/`` and rotate old ones.

    Args:
        model: The model to checkpoint.
        optimiser: Optimiser state dict.
        scheduler: Scheduler state dict.
        epoch: Current epoch index.
        metrics: Validation metrics at this checkpoint.
        cfg: Full project config.
        keep_last_n: Maximum number of recent checkpoints to retain.
    """
    latest_dir = Path(cfg.paths.latest_checkpoint_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = latest_dir / f"checkpoint_epoch{epoch:03d}.pt"
    model.save_checkpoint(
        path=ckpt_path,
        epoch=epoch,
        optimiser_state=optimiser.state_dict(),
        scheduler_state=scheduler.state_dict(),
        metrics={"top1": metrics.top1, "top5": metrics.top5, "loss": metrics.loss},
        cfg=cfg,
    )

    # Rotate old checkpoints
    if keep_last_n:
        existing = sorted(latest_dir.glob("checkpoint_epoch*.pt"))
        for old in existing[:-keep_last_n]:
            old.unlink()
            log.debug("Removed old checkpoint: %s", old)


def _save_best(
    model: SignModel,
    epoch: int,
    metrics: EpochMetrics,
    cfg: Config,
) -> None:
    """Save a checkpoint to ``trained_models/best/`` (overwrites previous best).

    Args:
        model: The model to checkpoint.
        epoch: Current epoch index.
        metrics: Validation metrics at this checkpoint.
        cfg: Full project config.
    """
    best_path = Path(cfg.paths.best_checkpoint_dir) / "checkpoint.pt"
    model.save_checkpoint(
        path=best_path,
        epoch=epoch,
        metrics={"top1": metrics.top1, "top5": metrics.top5, "loss": metrics.loss},
        cfg=cfg,
    )
    log.info(
        "New best checkpoint saved (top1=%.4f  top5=%.4f)",
        metrics.top1,
        metrics.top5,
    )


# =============================================================================
# Experiment logger helpers
# =============================================================================


def build_logger(logger_name: str, cfg: Config) -> object | None:
    """Initialise an experiment tracking logger.

    Args:
        logger_name: One of ``"wandb"``, ``"mlflow"``, ``"none"``.
        cfg: Full project config for run metadata.

    Returns:
        Logger object (W&B run or MLflow client), or ``None`` if disabled.
    """
    if logger_name == "wandb":
        try:
            import wandb  # type: ignore[import]

            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "wlasl300-sign-language"),
                entity=os.getenv("WANDB_ENTITY"),
                config=cfg.to_dict(),
                name=f"run_{time.strftime('%Y%m%d_%H%M%S')}",
            )
            log.info("W&B run initialised: %s", run.url)
            return run
        except Exception as exc:
            log.warning("W&B initialisation failed: %s — logging disabled", exc)
            return None

    if logger_name == "mlflow":
        try:
            import mlflow  # type: ignore[import]

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
            mlflow.set_experiment("wlasl300-sign-language")
            mlflow.start_run()
            mlflow.log_params(
                {k: v for k, v in cfg.to_dict().items() if isinstance(v, (int, float, str))}
            )
            log.info("MLflow run started")
            return mlflow
        except Exception as exc:
            log.warning("MLflow initialisation failed: %s — logging disabled", exc)
            return None

    return None


def _log_step(
    logger: object,
    loss_dict: dict[str, float],
    lr: float,
    step: int,
) -> None:
    """Log per-step metrics to the experiment tracker.

    Args:
        logger: W&B run or MLflow module.
        loss_dict: Dict from :meth:`~models.sign_model.SignModel.combined_loss`.
        lr: Current learning rate.
        step: Global training step.
    """
    payload = {**loss_dict, "lr": lr, "step": step}
    try:
        if hasattr(logger, "log"):  # W&B
            logger.log(payload, step=step)
        elif hasattr(logger, "log_metrics"):  # MLflow
            logger.log_metrics(payload, step=step)
    except Exception as exc:
        log.debug("Logger step error: %s", exc)


def _log_epoch(logger: object, metrics: EpochMetrics, epoch: int) -> None:
    """Log epoch-level metrics to the experiment tracker.

    Args:
        logger: W&B run or MLflow module.
        metrics: :class:`~utils.metrics.EpochMetrics` for this epoch.
        epoch: Epoch index.
    """
    payload = metrics.to_dict()
    try:
        if hasattr(logger, "log"):
            logger.log(payload, step=epoch)
        elif hasattr(logger, "log_metrics"):
            logger.log_metrics(payload, step=epoch)
    except Exception as exc:
        log.debug("Logger epoch error: %s", exc)


# =============================================================================
# Phase transition helper
# =============================================================================


def _get_phase(epoch: int, cfg: Config) -> int:
    """Determine the training phase for a given epoch.

    Args:
        epoch: Current epoch index (0-based).
        cfg: Full project config.

    Returns:
        Integer phase: 1, 2, or 3.
    """
    phase1_end = cfg.training.phase1.epochs
    phase2_end = phase1_end + cfg.training.phase2.epochs
    if epoch < phase1_end:
        return 1
    if epoch < phase2_end:
        return 2
    return 3


# =============================================================================
# Main training loop
# =============================================================================


def train(cfg: Config, resume_path: str | None = None) -> None:
    """Execute the full three-phase training pipeline.

    Sets up the model, dataloaders, optimiser, scheduler, and logger,
    then iterates over all epochs applying the correct training phase.
    Saves checkpoints to ``trained_models/`` and generates diagnostic
    plots to ``logs/plots/`` at the end of training.

    Args:
        cfg: Fully populated :class:`~config.base_config.Config` instance.
        resume_path: Optional path to a checkpoint to resume from.
            If provided, model weights, optimiser state, and the starting
            epoch are restored.
    """
    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    logging.basicConfig(
        level=getattr(logging, cfg.logging.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    set_seed(cfg.training.seed)
    device = get_device()
    cfg.paths.make_dirs()

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    log.info("Building dataloaders …")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # Load class embeddings and vocab for metrics and inference
    from utils.embedding_utils import load_embeddings_and_vocab

    class_embeddings, vocab = load_embeddings_and_vocab(
        cfg.paths.embeddings_file,
        cfg.paths.vocab_file,
        device=str(device),
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    start_epoch = 0

    if resume_path:
        log.info("Resuming from checkpoint: %s", resume_path)
        model, start_epoch, _ = SignModel.load_checkpoint(resume_path, cfg, device=str(device))
        start_epoch += 1  # start from the epoch after the saved one
    else:
        model = SignModel.from_config(cfg)
        model.to(device)

    log.info("\n%s", model.model_summary())

    # ------------------------------------------------------------------ #
    # Experiment logger
    # ------------------------------------------------------------------ #
    logger = build_logger(cfg.logging.logger, cfg)

    # ------------------------------------------------------------------ #
    # Early stopping state
    # ------------------------------------------------------------------ #
    es_cfg = cfg.early_stopping
    best_metric_value = float("-inf") if es_cfg.mode == "max" else float("inf")
    epochs_without_improvement = 0

    # ------------------------------------------------------------------ #
    # Training history (for plots)
    # ------------------------------------------------------------------ #
    train_losses: list[float] = []
    val_losses: list[float] = []
    top1_history: list[float] = []
    top5_history: list[float] = []
    throughput_history: list[float] = []

    # ------------------------------------------------------------------ #
    # Phase-aware training loop
    # ------------------------------------------------------------------ #
    total_epochs = cfg.training.epochs
    current_phase = -1
    optimiser: AdamW | None = None
    scheduler: CosineAnnealingWarmRestarts | None = None
    scaler = GradScaler(device.type, enabled=cfg.training.mixed_precision and device.type == "cuda")
    global_step = 0

    for epoch in range(start_epoch, total_epochs):
        phase = _get_phase(epoch, cfg)

        # Phase transition: rebuild optimiser and scheduler with new LRs
        if phase != current_phase:
            log.info("=" * 55)
            log.info("  Entering Phase %d (epoch %d)", phase, epoch + 1)
            log.info("=" * 55)

            model.apply_phase(phase, cfg)
            optimiser = build_optimiser(model, cfg, phase)
            scheduler = build_scheduler(optimiser, cfg)
            base_lrs = [g["lr"] for g in optimiser.param_groups]
            current_phase = phase

        warmup_steps = cfg.scheduler.warmup_steps if epoch == 0 else 0

        log.info(
            "Epoch %d/%d  phase=%d  lr=%.2e",
            epoch + 1,
            total_epochs,
            phase,
            optimiser.param_groups[-1]["lr"],
        )

        # ------------------------------------------------------------------ #
        # Train epoch
        # ------------------------------------------------------------------ #
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimiser=optimiser,
            scheduler=scheduler,
            scaler=scaler,
            class_embeddings=class_embeddings,
            cfg=cfg,
            epoch=epoch,
            global_step=global_step,
            warmup_steps=warmup_steps,
            base_lrs=base_lrs,
            logger=logger,
            device=device,
        )
        log.info("%s", train_metrics)

        # ------------------------------------------------------------------ #
        # Validation epoch
        # ------------------------------------------------------------------ #
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            class_embeddings=class_embeddings,
            cfg=cfg,
            epoch=epoch,
            split="val",
            device=device,
        )
        log.info("%s", val_metrics)

        # ------------------------------------------------------------------ #
        # Logging
        # ------------------------------------------------------------------ #
        if logger:
            _log_epoch(logger, train_metrics, epoch)
            _log_epoch(logger, val_metrics, epoch)

        # History for plots
        train_losses.append(train_metrics.loss)
        val_losses.append(val_metrics.loss)
        top1_history.append(val_metrics.top1)
        top5_history.append(val_metrics.top5)
        throughput_history.append(train_metrics.throughput_clips_per_sec)

        # ------------------------------------------------------------------ #
        # Checkpointing
        # ------------------------------------------------------------------ #
        if cfg.checkpointing.save_every_epoch:
            _save_latest(
                model,
                optimiser,
                scheduler,
                epoch,
                val_metrics,
                cfg,
                keep_last_n=cfg.checkpointing.keep_last_n,
            )

        current_metric = getattr(val_metrics, _metric_attr(es_cfg.monitor))
        is_best = (
            es_cfg.mode == "max" and current_metric > best_metric_value + es_cfg.min_delta
        ) or (es_cfg.mode == "min" and current_metric < best_metric_value - es_cfg.min_delta)

        if is_best:
            best_metric_value = current_metric
            epochs_without_improvement = 0
            if cfg.checkpointing.save_best:
                _save_best(model, epoch, val_metrics, cfg)
        else:
            epochs_without_improvement += 1
            log.info(
                "No improvement for %d/%d epochs (%s=%.4f  best=%.4f)",
                epochs_without_improvement,
                es_cfg.patience,
                es_cfg.monitor,
                current_metric,
                best_metric_value,
            )

        # ------------------------------------------------------------------ #
        # Early stopping check
        # ------------------------------------------------------------------ #
        if epochs_without_improvement >= es_cfg.patience:
            log.info(
                "Early stopping triggered at epoch %d " "(no improvement for %d epochs)",
                epoch + 1,
                es_cfg.patience,
            )
            break

    # ------------------------------------------------------------------ #
    # Post-training: test evaluation
    # ------------------------------------------------------------------ #
    if cfg.evaluation.run_test_eval:
        log.info("Running final test set evaluation …")
        best_ckpt = Path(cfg.paths.best_checkpoint_dir) / "checkpoint.pt"
        if best_ckpt.exists():
            model, _, _ = SignModel.load_checkpoint(best_ckpt, cfg, device=str(device))
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            class_embeddings=class_embeddings,
            cfg=cfg,
            epoch=total_epochs,
            split="test",
            device=device,
            compute_per_class=True,
            vocab=vocab,
        )
        log.info("%s", test_metrics)

        # Save per-class results
        per_class_path = Path(cfg.paths.log_dir) / "per_class_accuracy.json"
        with per_class_path.open("w") as f:
            json.dump(test_metrics.per_class_top1, f, indent=2)
        log.info("Per-class accuracy saved → %s", per_class_path)

        if test_metrics.per_class_top1:
            from utils.visualization import plot_per_class_accuracy

            plot_per_class_accuracy(
                test_metrics.per_class_top1,
                save_dir=cfg.paths.plots_dir,
            )

        if logger:
            _log_epoch(logger, test_metrics, total_epochs)

    # ------------------------------------------------------------------ #
    # Save training plots
    # ------------------------------------------------------------------ #
    plot_training_summary(
        train_losses=train_losses,
        val_losses=val_losses,
        top1_history=top1_history,
        top5_history=top5_history,
        throughput_history=throughput_history,
        save_dir=cfg.paths.plots_dir,
    )

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    if logger and hasattr(logger, "finish"):
        logger.finish()

    log.info("Training complete. Best %s = %.4f", es_cfg.monitor, best_metric_value)


# =============================================================================
# Helper: map monitor name → EpochMetrics attribute
# =============================================================================


def _metric_attr(monitor: str) -> str:
    """Map a metric monitor string to an :class:`~utils.metrics.EpochMetrics` attribute.

    Args:
        monitor: Metric name from config (e.g. ``"val_top1_accuracy"``).

    Returns:
        Attribute name on :class:`~utils.metrics.EpochMetrics`
        (e.g. ``"top1"``).
    """
    mapping = {
        "val_top1_accuracy": "top1",
        "val_top5_accuracy": "top5",
        "val_loss": "loss",
        "val_cosine_loss": "cosine_loss",
        "top1": "top1",
        "top5": "top5",
        "loss": "loss",
    }
    return mapping.get(monitor, "top1")


# =============================================================================
# CLI argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Returns:
        Parsed :class:`argparse.Namespace` with ``config``, ``resume``,
        and ``logger`` attributes, plus any dot-notation overrides collected
        in ``overrides``.
    """
    parser = argparse.ArgumentParser(
        description="Train the WLASL300 sign language recognition model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default=None,
        choices=["wandb", "mlflow", "none"],
        help="Override the experiment tracker set in config.yaml.",
    )
    return parser.parse_args()


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    """Entry point for the training script.

    Loads the config from ``--config``, applies CLI overrides, and calls
    :func:`train`.

    Called by ``uv run python train/train.py`` or the ``train`` console
    script defined in ``pyproject.toml``.
    """
    args = parse_args()

    cfg = Config.from_yaml(args.config)

    # Apply CLI logger override
    if args.logger and args.logger != "none":
        import dataclasses

        new_logging = dataclasses.replace(cfg.logging, logger=args.logger)
        cfg = dataclasses.replace(cfg, logging=new_logging)

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
