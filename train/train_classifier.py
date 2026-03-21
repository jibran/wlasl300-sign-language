"""Training script for the WLASL300 sign language classifier.

Trains :class:`~models.sign_model_classifier.SignModelClassifier` — an I3D
backbone with a fully-connected classification head — using cross-entropy loss
with label smoothing.

The same three-phase training schedule as ``train.py`` is used:

- **Phase 1** (epochs 1–10): Backbone frozen, head only.
- **Phase 2** (epochs 11–40): Last N backbone blocks unfrozen.
- **Phase 3** (epochs 41–60): Full backbone fine-tuning at low LR.

The same pre-processed WLASL300 dataset is used — no changes to the data
pipeline are required.

Usage::

    uv run python train/train_classifier.py --config config/config.yaml

    # Resume from checkpoint
    uv run python train/train_classifier.py \\
        --config config/config.yaml \\
        --resume trained_models/classifier/latest/checkpoint.pt

    # Override batch size
    uv run python train/train_classifier.py \\
        --config config/config.yaml \\
        --batch_size 16
"""

from __future__ import annotations

import argparse
import json
import logging
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

from config.base_config import Config
from dataset.data.wlasl_dataset import build_dataloaders
from models.sign_model_classifier import SignModelClassifier
from models.sign_model_linear import SignModelLinear
from utils.visualization import plot_accuracy_curves, plot_loss_curves, plot_throughput

log = logging.getLogger(__name__)

AnyClassifier = SignModelClassifier | SignModelLinear

# Checkpoint directories (separate from the embedding model)
# Checkpoint dirs are set dynamically based on --model flag
_BEST_DIR = "trained_models/classifier/best"
_LATEST_DIR = "trained_models/classifier/latest"


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducible training.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info("Random seed set to %d", seed)


# =============================================================================
# Device
# =============================================================================


def get_device() -> torch.device:
    """Return the best available compute device.

    Returns:
        ``torch.device("cuda")`` or ``torch.device("cpu")``.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(
            "GPU: %s  VRAM: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        device = torch.device("cpu")
        log.warning("No GPU found — training on CPU will be very slow")
    return device


# =============================================================================
# Phase helpers
# =============================================================================


def _get_phase(epoch: int, cfg: Config) -> int:
    """Map a 0-based epoch index to a training phase (1, 2, or 3).

    Args:
        epoch: 0-based epoch index.
        cfg: Project config.

    Returns:
        Integer phase 1, 2, or 3.
    """
    p1 = cfg.training.phase1.epochs
    p2 = cfg.training.phase2.epochs
    if epoch < p1:
        return 1
    if epoch < p1 + p2:
        return 2
    return 3


# =============================================================================
# Optimiser and scheduler
# =============================================================================


def build_optimiser(
    model: AnyClassifier,
    cfg: Config,
    phase: int,
) -> AdamW:
    """Build an AdamW optimiser with phase-appropriate learning rates.

    The backbone and head use separate parameter groups so the backbone can
    be trained at a lower LR than the head during phases 2 and 3.

    Args:
        model: The classifier model.
        cfg: Project config.
        phase: Current training phase (1, 2, or 3).

    Returns:
        Configured :class:`~torch.optim.AdamW` optimiser.
    """
    phase_cfg = getattr(cfg.training, f"phase{phase}")
    lr = phase_cfg.learning_rate
    opt_cfg = cfg.optimiser

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())

    if backbone_params:
        param_groups = [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr},
        ]
    else:
        param_groups = [{"params": head_params, "lr": lr}]

    optimiser = AdamW(
        param_groups,
        weight_decay=opt_cfg.weight_decay,
        betas=tuple(opt_cfg.betas),
        eps=opt_cfg.eps,
    )
    log.info(
        "Optimiser: AdamW  phase=%d  lr=%.2e  trainable_params=%d",
        phase,
        lr,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return optimiser


def build_scheduler(
    optimiser: AdamW,
    cfg: Config,
) -> CosineAnnealingWarmRestarts:
    """Build a cosine annealing LR scheduler with warm restarts.

    Args:
        optimiser: The AdamW optimiser.
        cfg: Project config.

    Returns:
        Configured :class:`~torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.
    """
    sched_cfg = cfg.scheduler
    return CosineAnnealingWarmRestarts(
        optimiser,
        T_0=sched_cfg.T_0,
        T_mult=sched_cfg.T_mult,
        eta_min=sched_cfg.min_lr,
    )


# =============================================================================
# Warmup
# =============================================================================


def linear_warmup(
    optimiser: AdamW,
    step: int,
    warmup_steps: int,
    base_lrs: list[float],
) -> None:
    """Scale LRs linearly from 0 to their base values over warmup steps.

    Args:
        optimiser: Optimiser whose parameter group LRs to scale.
        step: Current global training step (0-indexed).
        warmup_steps: Number of warmup steps.
        base_lrs: Target LR per parameter group.
    """
    if warmup_steps <= 0 or step >= warmup_steps:
        return
    scale = (step + 1) / warmup_steps
    for group, base_lr in zip(optimiser.param_groups, base_lrs, strict=True):
        group["lr"] = base_lr * scale


# =============================================================================
# Accuracy helpers
# =============================================================================


def topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Compute top-k accuracy from logits and integer labels.

    Args:
        logits: Raw logit tensor ``(B, num_classes)``.
        labels: Ground-truth class indices ``(B,)``, int64.
        topk: k-values to evaluate.

    Returns:
        Dict mapping ``"top{k}"`` to float accuracy in ``[0, 1]``.
    """
    results: dict[str, float] = {}
    num_classes = logits.shape[1]
    for k in topk:
        k_eff = min(k, num_classes)
        _, pred_indices = logits.topk(k_eff, dim=-1, largest=True)
        correct = pred_indices.eq(labels.unsqueeze(1).expand_as(pred_indices))
        results[f"top{k}"] = correct.any(dim=1).float().mean().item()
    return results


# =============================================================================
# Train and eval loops
# =============================================================================


def train_one_epoch(
    model: AnyClassifier,
    loader: DataLoader,
    optimiser: AdamW,
    scheduler: CosineAnnealingWarmRestarts,
    scaler: GradScaler,
    cfg: Config,
    epoch: int,
    global_step: int,
    warmup_steps: int,
    base_lrs: list[float],
    device: torch.device,
    label_smoothing: float = 0.1,
) -> tuple[dict[str, float], int]:
    """Run one full training epoch.

    Args:
        model: Classifier in training mode.
        loader: Training dataloader yielding ``(video, embedding, label_idx)``.
        optimiser: AdamW optimiser.
        scheduler: LR scheduler.
        scaler: AMP GradScaler.
        cfg: Project config.
        epoch: Current epoch (0-based).
        global_step: Global step counter carried across epochs.
        warmup_steps: LR warmup steps.
        base_lrs: Base LR per parameter group for warmup.
        device: Compute device.
        label_smoothing: Cross-entropy label smoothing factor.

    Returns:
        Tuple of metric dict and updated global step.
    """
    model.train()
    model.backbone.set_train_mode(is_train=True)

    accum_steps = cfg.training.grad_accumulation_steps
    use_amp = cfg.training.mixed_precision and device.type == "cuda"

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    n_samples = 0
    t0 = time.perf_counter()

    optimiser.zero_grad()

    # DataLoader yields (video, word2vec_embedding, label_idx).
    # We only use video and label_idx here — the embedding is ignored.
    for batch_idx, (videos, _embeddings, labels) in enumerate(loader):
        videos = videos.to(device, non_blocking=True)  # (B, 3, T, H, W)
        labels = labels.to(device, non_blocking=True)  # (B,)

        linear_warmup(optimiser, global_step, warmup_steps, base_lrs)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(videos)  # (B, num_classes)
            loss = SignModelClassifier.loss(logits, labels, label_smoothing=label_smoothing)
            scaled_loss = loss / accum_steps

        scaler.scale(scaled_loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            if cfg.training.grad_clip_norm:
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()
            scheduler.step(epoch + batch_idx / len(loader))
            global_step += 1

        with torch.no_grad():
            acc = topk_accuracy(logits, labels, topk=(1, 5))

        b = videos.shape[0]
        total_loss += loss.item() * b
        total_top1 += acc["top1"] * b
        total_top5 += acc["top5"] * b
        n_samples += b

    elapsed = time.perf_counter() - t0
    metrics = {
        "loss": total_loss / max(n_samples, 1),
        "top1": total_top1 / max(n_samples, 1),
        "top5": total_top5 / max(n_samples, 1),
        "throughput_clips_per_sec": n_samples / max(elapsed, 1e-6),
        "n_samples": n_samples,
    }
    return metrics, global_step


@torch.no_grad()
def evaluate(
    model: AnyClassifier,
    loader: DataLoader,
    cfg: Config,
    epoch: int,
    split: str,
    device: torch.device,
    label_smoothing: float = 0.1,
) -> dict[str, float]:
    """Evaluate the classifier on a val or test split.

    Args:
        model: Classifier in eval mode.
        loader: Val or test dataloader.
        cfg: Project config.
        epoch: Current epoch (0-based).
        split: ``"val"`` or ``"test"``.
        device: Compute device.
        label_smoothing: Cross-entropy label smoothing factor.

    Returns:
        Dict of metric values.
    """
    model.eval()
    use_amp = cfg.training.mixed_precision and device.type == "cuda"

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    n_samples = 0

    for videos, _embeddings, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(videos)
            loss = SignModelClassifier.loss(logits, labels, label_smoothing=label_smoothing)

        acc = topk_accuracy(logits, labels, topk=(1, 5))
        b = videos.shape[0]
        total_loss += loss.item() * b
        total_top1 += acc["top1"] * b
        total_top5 += acc["top5"] * b
        n_samples += b

    metrics = {
        "loss": total_loss / max(n_samples, 1),
        "top1": total_top1 / max(n_samples, 1),
        "top5": total_top5 / max(n_samples, 1),
        "n_samples": n_samples,
    }

    log.info(
        "[%s epoch=%d] loss=%.4f  top1=%.3f  top5=%.3f",
        split,
        epoch + 1,
        metrics["loss"],
        metrics["top1"],
        metrics["top5"],
    )
    return metrics


# =============================================================================
# Checkpointing helpers
# =============================================================================


def _save_best(
    model: AnyClassifier,
    epoch: int,
    metrics: dict[str, float],
    cfg: Config,
    best_dir: str = _BEST_DIR,
) -> None:
    """Save the best checkpoint to ``trained_models/classifier/best/``.

    Args:
        model: Classifier model.
        epoch: Current epoch (0-based).
        metrics: Validation metrics dict.
        cfg: Project config.
    """
    path = Path(best_dir) / "checkpoint.pt"
    model.save_checkpoint(
        path=path,
        epoch=epoch,
        metrics=metrics,
        cfg=cfg,
    )
    meta_path = Path(best_dir) / "best_metrics.json"
    meta_path.write_text(
        json.dumps({"epoch": epoch + 1, **metrics}, indent=2),
        encoding="utf-8",
    )
    log.info("New best checkpoint  top1=%.4f  saved → %s", metrics.get("top1", 0), path)


def _save_latest(
    model: AnyClassifier,
    optimiser: AdamW,
    scheduler: CosineAnnealingWarmRestarts,
    epoch: int,
    metrics: dict[str, float],
    cfg: Config,
    latest_dir: str = _LATEST_DIR,
) -> None:
    """Save the latest checkpoint to ``trained_models/classifier/latest/``.

    Args:
        model: Classifier model.
        optimiser: Current optimiser.
        scheduler: Current scheduler.
        epoch: Current epoch (0-based).
        metrics: Validation metrics dict.
        cfg: Project config.
    """
    path = Path(latest_dir) / "checkpoint.pt"
    model.save_checkpoint(
        path=path,
        epoch=epoch,
        optimiser_state=optimiser.state_dict(),
        scheduler_state=scheduler.state_dict(),
        metrics=metrics,
        cfg=cfg,
    )


# =============================================================================
# Main training loop
# =============================================================================


def train(cfg: Config, resume_path: str | None = None, model_type: str = "classifier") -> None:
    """Run the full three-phase classifier training pipeline.

    Args:
        cfg: Fully populated project config.
        resume_path: Optional path to a checkpoint to resume from.
        model_type: ``"classifier"`` (deep FC head) or ``"linear"`` (single linear layer).
    """
    logging.basicConfig(
        level=getattr(logging, cfg.logging.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    set_seed(cfg.training.seed)
    device = get_device()

    best_dir = f"trained_models/{model_type}/best"
    latest_dir = f"trained_models/{model_type}/latest"
    log_dir = f"logs/{model_type}"
    Path(best_dir).mkdir(parents=True, exist_ok=True)
    Path(latest_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    log.info("Building dataloaders …")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    start_epoch = 0

    ModelClass = SignModelLinear if model_type == "linear" else SignModelClassifier
    if resume_path:
        log.info("Resuming from checkpoint: %s", resume_path)
        model, start_epoch, _ = ModelClass.load_checkpoint(resume_path, cfg, device=str(device))
        start_epoch += 1
    else:
        model = ModelClass.from_config(cfg)
        model.to(device)

    log.info("\n%s", model.model_summary())

    # ------------------------------------------------------------------ #
    # Early stopping state
    # ------------------------------------------------------------------ #
    es_cfg = cfg.early_stopping
    best_top1 = 0.0
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
    scaler = GradScaler(
        device.type,
        enabled=cfg.training.mixed_precision and device.type == "cuda",
    )
    global_step = 0

    label_smoothing = 0.1

    for epoch in range(start_epoch, total_epochs):
        phase = _get_phase(epoch, cfg)

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

        # Train
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimiser=optimiser,
            scheduler=scheduler,
            scaler=scaler,
            cfg=cfg,
            epoch=epoch,
            global_step=global_step,
            warmup_steps=warmup_steps,
            base_lrs=base_lrs,
            device=device,
            label_smoothing=label_smoothing,
        )
        log.info(
            "[train epoch=%d] loss=%.4f  top1=%.3f  top5=%.3f  thr=%.1f clips/s",
            epoch + 1,
            train_metrics["loss"],
            train_metrics["top1"],
            train_metrics["top5"],
            train_metrics["throughput_clips_per_sec"],
        )

        # Validate
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            cfg=cfg,
            epoch=epoch,
            split="val",
            device=device,
            label_smoothing=label_smoothing,
        )

        # History
        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])
        top1_history.append(val_metrics["top1"])
        top5_history.append(val_metrics["top5"])
        throughput_history.append(train_metrics["throughput_clips_per_sec"])

        log.info(
            "  train top1=%.3f  top5=%.3f  |  val top1=%.3f  top5=%.3f",
            train_metrics["top1"],
            train_metrics["top5"],
            val_metrics["top1"],
            val_metrics["top5"],
        )

        # Checkpoint latest
        _save_latest(model, optimiser, scheduler, epoch, val_metrics, cfg, latest_dir=latest_dir)

        # Checkpoint best
        val_top1 = val_metrics["top1"]
        if val_top1 > best_top1 + es_cfg.min_delta:
            best_top1 = val_top1
            epochs_without_improvement = 0
            _save_best(model, epoch, val_metrics, cfg, best_dir=best_dir)
        else:
            epochs_without_improvement += 1
            log.info(
                "No improvement for %d/%d epochs  (best top1=%.4f)",
                epochs_without_improvement,
                es_cfg.patience,
                best_top1,
            )

        if es_cfg.enabled and epochs_without_improvement >= es_cfg.patience and phase == 3:
            log.info(
                "Early stopping triggered at epoch %d  (best top1=%.4f)",
                epoch + 1,
                best_top1,
            )
            break

    # ------------------------------------------------------------------ #
    # Final test evaluation
    # ------------------------------------------------------------------ #
    log.info("Running final test evaluation …")
    best_ckpt = Path(best_dir) / "checkpoint.pt"
    if best_ckpt.exists():
        model, _, _ = ModelClass.load_checkpoint(str(best_ckpt), cfg, device=str(device))
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        cfg=cfg,
        epoch=total_epochs - 1,
        split="test",
        device=device,
    )
    log.info(
        "Test  top1=%.4f  top5=%.4f",
        test_metrics["top1"],
        test_metrics["top5"],
    )

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    plot_loss_curves(train_losses, val_losses, save_dir=log_dir)
    plot_accuracy_curves(top1_history, top5_history, save_dir=log_dir)
    if throughput_history:
        plot_throughput(throughput_history, save_dir=log_dir)
    log.info("Training complete. Plots saved to %s", log_dir)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    p = argparse.ArgumentParser(
        description="Train WLASL300 sign language classifier (cross-entropy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to classifier checkpoint to resume training from.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override config batch size.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total training epochs.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="classifier",
        choices=["classifier", "linear"],
        help='Head type: "classifier" (deep FC) or "linear" (single layer).',
    )
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Cross-entropy label smoothing factor.",
    )
    p.add_argument(
        "--phase1_epochs",
        type=int,
        default=None,
        help="Override phase 1 epoch count (backbone frozen).",
    )
    p.add_argument(
        "--phase2_epochs",
        type=int,
        default=None,
        help="Override phase 2 epoch count (last N blocks unfrozen).",
    )
    p.add_argument(
        "--phase3_epochs",
        type=int,
        default=None,
        help="Override phase 3 epoch count (full unfreeze).",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for the classifier training script."""
    import dataclasses

    args = parse_args()
    cfg = Config.from_yaml(args.config)

    if args.batch_size is not None:
        new_training = dataclasses.replace(cfg.training, batch_size=args.batch_size)
        cfg = dataclasses.replace(cfg, training=new_training)

    # Apply phase epoch overrides
    if any(x is not None for x in (args.phase1_epochs, args.phase2_epochs, args.phase3_epochs)):
        p1 = dataclasses.replace(
            cfg.training.phase1,
            epochs=(
                args.phase1_epochs if args.phase1_epochs is not None else cfg.training.phase1.epochs
            ),
        )
        p2 = dataclasses.replace(
            cfg.training.phase2,
            epochs=(
                args.phase2_epochs if args.phase2_epochs is not None else cfg.training.phase2.epochs
            ),
        )
        p3 = dataclasses.replace(
            cfg.training.phase3,
            epochs=(
                args.phase3_epochs if args.phase3_epochs is not None else cfg.training.phase3.epochs
            ),
        )
        new_training = dataclasses.replace(cfg.training, phase1=p1, phase2=p2, phase3=p3)
        cfg = dataclasses.replace(cfg, training=new_training)

    # Apply total epoch override (must come after phase overrides)
    if args.epochs is not None:
        new_training = dataclasses.replace(cfg.training, epochs=args.epochs)
        cfg = dataclasses.replace(cfg, training=new_training)

    train(cfg, resume_path=args.resume, model_type=args.model)


if __name__ == "__main__":
    main()
