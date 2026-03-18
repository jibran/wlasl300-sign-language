"""Training and evaluation metrics for sign language recognition.

This module provides stateful metric accumulators and stateless metric
functions used throughout the training loop and evaluation scripts.

:class:`MetricTracker` accumulates predictions and ground-truth labels
across batches and computes final epoch-level metrics in one call.

:class:`EpochMetrics` is a lightweight dataclass that the training loop
populates and passes to the logger and visualizer.

Example::

    from utils.metrics import MetricTracker

    tracker = MetricTracker(num_classes=300, topk=(1, 5))

    for videos, embeddings, labels in val_loader:
        pred_emb = model(videos)
        tracker.update(pred_emb, class_matrix, labels)

    results = tracker.compute()
    print(results)
    # {"top1": 0.621, "top5": 0.883, "mean_cosine_sim": 0.74, ...}

    tracker.reset()  # clear for next epoch
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

log = logging.getLogger(__name__)


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass
class EpochMetrics:
    """Metrics computed over one full epoch.

    Attributes:
        split: Name of the data split (``"train"``, ``"val"``, ``"test"``).
        epoch: Epoch index (0-based).
        loss: Mean total loss over the epoch.
        cosine_loss: Mean cosine component of the loss.
        triplet_loss: Mean triplet component of the loss (0 if disabled).
        top1: Top-1 nearest-neighbour accuracy in ``[0, 1]``.
        top5: Top-5 nearest-neighbour accuracy in ``[0, 1]``.
        mean_cosine_sim: Mean cosine similarity between predicted and target
            embeddings.
        num_samples: Total number of video clips processed this epoch.
        throughput_clips_per_sec: Training throughput in video clips per second.
        per_class_top1: Optional dict mapping class label to top-1 accuracy.
            Populated only when ``compute_per_class=True`` is passed to
            :meth:`MetricTracker.compute`.
    """

    split: str = "val"
    epoch: int = 0
    loss: float = 0.0
    cosine_loss: float = 0.0
    triplet_loss: float = 0.0
    top1: float = 0.0
    top5: float = 0.0
    mean_cosine_sim: float = 0.0
    num_samples: int = 0
    throughput_clips_per_sec: float = 0.0
    per_class_top1: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int]:
        """Serialise to a flat dict suitable for logging to W&B or MLflow.

        Returns:
            Dict with metric name keys and float/int values.
        """
        return {
            "loss": self.loss,
            "cosine_loss": self.cosine_loss,
            "triplet_loss": self.triplet_loss,
            "top1": self.top1,
            "top5": self.top5,
            "mean_cosine_sim": self.mean_cosine_sim,
            "num_samples": self.num_samples,
            "throughput_clips_per_sec": self.throughput_clips_per_sec,
        }

    def __str__(self) -> str:
        """Return a concise one-line summary string.

        Returns:
            Human-readable metric summary string.
        """
        return (
            f"[{self.split} epoch={self.epoch}] "
            f"loss={self.loss:.4f}  "
            f"top1={self.top1:.3f}  "
            f"top5={self.top5:.3f}  "
            f"cosine_sim={self.mean_cosine_sim:.3f}  "
            f"thr={self.throughput_clips_per_sec:.1f} clips/s"
        )


# =============================================================================
# Stateful accumulator
# =============================================================================


class MetricTracker:
    """Accumulates per-batch predictions and computes epoch-level metrics.

    Thread-safe for single-process training.  For multi-GPU / DDP training,
    call :meth:`reset` at the start of each epoch on each rank and aggregate
    with ``torch.distributed.all_reduce`` before calling :meth:`compute`.

    Args:
        num_classes: Total number of sign language classes.
        topk: Tuple of k-values for top-k accuracy computation.

    Example::

        tracker = MetricTracker(num_classes=300, topk=(1, 5))
        for batch in loader:
            pred_emb, labels = model(batch)
            tracker.update(pred_emb, class_matrix, labels, loss_dict)
        metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(
        self,
        num_classes: int = 300,
        topk: tuple[int, ...] = (1, 5),
    ) -> None:
        """Initialise the tracker.

        Args:
            num_classes: Number of classes in the dataset.
            topk: k-values for top-k accuracy.
        """
        self._num_classes = num_classes
        self._topk = topk
        self._max_k = max(topk)
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated state.  Call at the start of each epoch."""
        self._all_preds: list[Tensor] = []
        self._all_targets: list[Tensor] = []
        self._all_labels: list[Tensor] = []
        self._loss_sum: float = 0.0
        self._cosine_loss_sum: float = 0.0
        self._triplet_loss_sum: float = 0.0
        self._num_batches: int = 0
        self._num_samples: int = 0

    def update(
        self,
        pred_embeddings: Tensor,
        class_embeddings: Tensor,
        true_label_indices: Tensor,
        loss_dict: dict[str, float] | None = None,
    ) -> None:
        """Accumulate one batch of predictions and ground-truth labels.

        Args:
            pred_embeddings: Predicted embeddings ``(B, D)``, L2-normalised,
                on any device.
            class_embeddings: Class embedding matrix ``(C, D)``, L2-normalised.
            true_label_indices: Ground-truth class indices ``(B,)``, int64.
            loss_dict: Optional dict from
                :meth:`~models.sign_model.SignModel.combined_loss` with keys
                ``"cosine_loss"``, ``"triplet_loss"``, ``"total_loss"``.
        """
        # Move to CPU for accumulation — avoids filling GPU memory
        pred_cpu = pred_embeddings.detach().cpu()
        class_cpu = class_embeddings.detach().cpu()
        labels_cpu = true_label_indices.detach().cpu()

        self._all_preds.append(pred_cpu)
        self._all_targets.append(class_cpu[labels_cpu])  # (B, D)
        self._all_labels.append(labels_cpu)

        B = pred_cpu.shape[0]
        self._num_samples += B
        self._num_batches += 1

        if loss_dict:
            self._loss_sum += loss_dict.get("total_loss", 0.0)
            self._cosine_loss_sum += loss_dict.get("cosine_loss", 0.0)
            self._triplet_loss_sum += loss_dict.get("triplet_loss", 0.0)

    def compute(
        self,
        compute_per_class: bool = False,
        vocab: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute all epoch-level metrics from accumulated predictions.

        Args:
            compute_per_class: If ``True``, compute top-1 accuracy for each
                class individually.  Requires ``vocab`` to map indices to
                label strings.
            vocab: Class name list for per-class accuracy labelling.  Required
                when ``compute_per_class=True``.

        Returns:
            Dict with metric names as keys and float values:
            ``top1``, ``top5``, ``mean_cosine_sim``, ``loss``,
            ``cosine_loss``, ``triplet_loss``, and optionally
            ``per_class/<label>`` entries.

        Raises:
            RuntimeError: If called before any :meth:`update` calls.
        """
        if not self._all_preds:
            raise RuntimeError(
                "MetricTracker.compute() called with no accumulated data. "
                "Call update() at least once before compute()."
            )

        all_preds = torch.cat(self._all_preds, dim=0)  # (N, D)
        all_targets = torch.cat(self._all_targets, dim=0)  # (N, D)
        all_labels = torch.cat(self._all_labels, dim=0)  # (N,)

        # Build class embedding matrix from unique label indices
        # (avoids needing to pass class_embeddings separately)
        C = int(all_labels.max().item()) + 1

        # Reconstruct partial class matrix from seen labels
        D = all_preds.shape[1]
        class_matrix = torch.zeros(C, D)
        for i, (_pred, lbl) in enumerate(zip(all_preds, all_labels, strict=True)):
            class_matrix[lbl.item()] = all_targets[i]

        # Similarities: (N, C)
        similarities = all_preds @ class_matrix.T

        # Top-k accuracy
        results: dict[str, float] = {}
        for k in self._topk:
            _, top_indices = similarities.topk(k=min(k, C), dim=-1, largest=True)
            true_expanded = all_labels.unsqueeze(1).expand_as(top_indices)
            correct = top_indices.eq(true_expanded).any(dim=1).float()
            results[f"top{k}"] = correct.mean().item()

        # Mean cosine similarity between predicted and true target embedding
        cos_sim = torch.nn.functional.cosine_similarity(all_preds, all_targets, dim=-1)
        results["mean_cosine_sim"] = cos_sim.mean().item()

        # Loss averages
        if self._num_batches > 0:
            results["loss"] = self._loss_sum / self._num_batches
            results["cosine_loss"] = self._cosine_loss_sum / self._num_batches
            results["triplet_loss"] = self._triplet_loss_sum / self._num_batches
        else:
            results.update({"loss": 0.0, "cosine_loss": 0.0, "triplet_loss": 0.0})

        # Per-class top-1
        if compute_per_class:
            predicted = similarities.argmax(dim=-1)  # (N,)
            class_correct: dict[int, list[int]] = defaultdict(list)
            for pred_lbl, true_lbl in zip(predicted.tolist(), all_labels.tolist(), strict=True):
                class_correct[true_lbl].append(int(pred_lbl == true_lbl))

            for lbl_idx, corrects in class_correct.items():
                name = vocab[lbl_idx] if vocab else str(lbl_idx)
                results[f"per_class/{name}"] = float(np.mean(corrects))

        return results

    @property
    def num_samples(self) -> int:
        """Total number of samples accumulated since last :meth:`reset`.

        Returns:
            Integer sample count.
        """
        return self._num_samples


# =============================================================================
# Stateless helpers
# =============================================================================


def compute_topk_accuracy(
    pred_embeddings: Tensor,
    class_embeddings: Tensor,
    true_label_indices: Tensor,
    topk: tuple[int, ...] = (1, 5),
    k: int | None = None,
) -> dict[str, float] | float:
    """Compute top-k nearest-neighbour accuracy.

    Two calling conventions are supported:

    - ``compute_topk_accuracy(pred, class_emb, labels, topk=(1, 5))``
      returns a ``dict`` mapping ``"top1"`` / ``"top5"`` to float accuracy.
    - ``compute_topk_accuracy(pred, class_emb, labels, k=1)``
      returns a single ``float`` accuracy for that k value.

    Args:
        pred_embeddings: Predicted embeddings ``(B, D)``, L2-normalised.
        class_embeddings: Class embedding matrix ``(C, D)``, L2-normalised.
        true_label_indices: Ground-truth class indices ``(B,)``, int64.
        topk: Tuple of k-values.  Used when ``k`` is not provided.
        k: Single k-value.  When provided, overrides ``topk`` and a single
            float is returned instead of a dict.

    Returns:
        Float accuracy when ``k`` is given; dict of accuracies otherwise.
    """
    similarities = pred_embeddings @ class_embeddings.T  # (B, C)
    C = class_embeddings.shape[0]

    if k is not None:
        k_eff = min(k, C)
        _, top_indices = similarities.topk(k=k_eff, dim=-1, largest=True)
        true_exp = true_label_indices.unsqueeze(1).expand_as(top_indices)
        correct = top_indices.eq(true_exp).any(dim=1).float()
        return correct.mean().item()

    results: dict[str, float] = {}
    for ki in topk:
        k_eff = min(ki, C)
        _, top_indices = similarities.topk(k=k_eff, dim=-1, largest=True)
        true_exp = true_label_indices.unsqueeze(1).expand_as(top_indices)
        correct = top_indices.eq(true_exp).any(dim=1).float()
        results[f"top{ki}"] = correct.mean().item()

    return results


def mean_cosine_similarity(pred: Tensor, target: Tensor) -> float:
    """Compute mean cosine similarity between two matched embedding batches.

    Args:
        pred: Predicted embeddings ``(B, D)``, L2-normalised.
        target: Target embeddings ``(B, D)``, L2-normalised.

    Returns:
        Mean cosine similarity as a Python float.
    """
    return torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()


def throughput(num_samples: int, elapsed_secs: float) -> float:
    """Compute training throughput in samples per second.

    Args:
        num_samples: Number of video clips processed.
        elapsed_secs: Wall-clock time elapsed in seconds.

    Returns:
        Float samples per second (0.0 if elapsed_secs is 0).
    """
    if elapsed_secs <= 0:
        return 0.0
    return num_samples / elapsed_secs
