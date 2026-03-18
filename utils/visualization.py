"""Visualization utilities for training diagnostics and result analysis.

This module provides plotting functions for:

- Training and validation loss curves over epochs.
- Top-1 / Top-5 accuracy curves over epochs.
- Training throughput (video clips / second) over epochs.
- Per-class accuracy bar chart (identifying the hardest classes).
- 2-D t-SNE scatter of predicted embedding space coloured by class.
- Cosine similarity distribution between predicted and target embeddings.

All functions save figures to ``save_dir`` as ``.png`` files and optionally
return the :class:`matplotlib.figure.Figure` object for programmatic use
(e.g. logging to W&B).

Example::

    from utils.visualization import (
        plot_loss_curves, plot_accuracy_curves, plot_throughput,
        plot_per_class_accuracy, plot_embedding_scatter,
    )

    plot_loss_curves(train_losses, val_losses, save_dir="logs/plots/")
    plot_accuracy_curves(top1_history, top5_history, save_dir="logs/plots/")
    plot_throughput(throughput_history, save_dir="logs/plots/")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# =============================================================================
# Shared helpers
# =============================================================================


def _get_axes(
    figsize: tuple[float, float] = (9, 5),
) -> tuple:
    """Create and return a new figure and axes with a clean style.

    Args:
        figsize: Figure size in inches ``(width, height)``.

    Returns:
        Tuple of ``(fig, ax)`` from ``plt.subplots()``.
    """
    import matplotlib.pyplot as plt  # type: ignore[import]

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
        }
    )
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _save(fig: object, save_dir: str | Path, filename: str) -> Path:
    """Save a figure to ``save_dir / filename`` and close it.

    Args:
        fig: Matplotlib figure object.
        save_dir: Directory to save the figure in.
        filename: File name including extension (e.g. ``"loss.png"``).

    Returns:
        :class:`Path` to the saved file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename
    fig.savefig(out_path, dpi=120, bbox_inches="tight")

    import matplotlib.pyplot as plt

    plt.close(fig)

    log.info("Saved plot → %s", out_path)
    return out_path


# =============================================================================
# Loss curves
# =============================================================================


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_dir: str | Path = "logs/plots",
    filename: str = "loss_curves.png",
    return_fig: bool = False,
) -> object | None:
    """Plot training and validation loss curves over epochs.

    Args:
        train_losses: List of mean training loss values, one per epoch.
        val_losses: List of mean validation loss values, one per epoch.
        save_dir: Directory to save the figure.
        filename: Output file name.
        return_fig: If ``True``, return the figure instead of closing it.
            Useful for logging to W&B via ``wandb.Image(fig)``.

    Returns:
        :class:`~pathlib.Path` to the saved PNG, or the figure if ``return_fig=True``.
    """

    fig, ax = _get_axes()
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train", color="#534AB7", linewidth=2)
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        ax.plot(
            val_epochs,
            val_losses,
            label="Val",
            color="#1D9E75",
            linewidth=2,
            linestyle="--",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (cosine embedding)")
    ax.set_title("Training and validation loss")
    ax.legend()

    out = _save(fig, save_dir, filename)
    if return_fig:
        return fig
    return out


# =============================================================================
# Accuracy curves
# =============================================================================


def plot_accuracy_curves(
    top1_history: list[float],
    top5_history: list[float],
    save_dir: str | Path = "logs/plots",
    filename: str = "accuracy_curves.png",
    return_fig: bool = False,
) -> object | None:
    """Plot top-1 and top-5 accuracy over training epochs.

    Args:
        top1_history: List of validation top-1 accuracy values per epoch.
        top5_history: List of validation top-5 accuracy values per epoch.
        save_dir: Directory to save the figure.
        filename: Output file name.
        return_fig: Return the figure object instead of closing it.

    Returns:
        :class:`~pathlib.Path` to the saved PNG, or the figure if ``return_fig=True``.
    """
    fig, ax = _get_axes()
    if top1_history:
        ax.plot(
            range(1, len(top1_history) + 1),
            top1_history,
            label="Top-1",
            color="#534AB7",
            linewidth=2,
        )
    if top5_history:
        ax.plot(
            range(1, len(top5_history) + 1),
            top5_history,
            label="Top-5",
            color="#D85A30",
            linewidth=2,
            linestyle="--",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(
        __import__("matplotlib.ticker", fromlist=["PercentFormatter"]).PercentFormatter(xmax=1)
    )
    ax.set_title("Validation accuracy (nearest-neighbour retrieval)")
    ax.legend()

    out = _save(fig, save_dir, filename)
    if return_fig:
        return fig
    return out


# =============================================================================
# Throughput plot
# =============================================================================


def plot_throughput(
    throughput_history: list[float],
    save_dir: str | Path = "logs/plots",
    filename: str = "throughput.png",
    return_fig: bool = False,
) -> object | None:
    """Plot training throughput (video clips / second) over epochs.

    Throughput drops are a useful diagnostic for data loading bottlenecks
    and GPU memory pressure.

    Args:
        throughput_history: List of throughput values (clips/sec) per epoch.
        save_dir: Directory to save the figure.
        filename: Output file name.
        return_fig: Return the figure object instead of closing it.

    Returns:
        :class:`~pathlib.Path` to the saved PNG, or the figure if ``return_fig=True``.
    """
    if not throughput_history:

        fig, ax = _get_axes()
        ax.set_title("Training throughput per epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Throughput (clips / second)")
        out = _save(fig, save_dir, filename)
        return fig if return_fig else out

    fig, ax = _get_axes()
    epochs = range(1, len(throughput_history) + 1)

    ax.bar(
        epochs,
        throughput_history,
        color="#1D9E75",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Throughput (clips / second)")
    ax.set_title("Training throughput per epoch")

    # Annotate mean throughput
    mean_thr = float(np.mean(throughput_history))
    ax.axhline(
        mean_thr,
        color="#534AB7",
        linewidth=1.5,
        linestyle="--",
        label=f"Mean: {mean_thr:.1f} clips/s",
    )
    ax.legend()

    out = _save(fig, save_dir, filename)
    if return_fig:
        return fig
    return out


# =============================================================================
# Per-class accuracy
# =============================================================================


def plot_per_class_accuracy(
    per_class_accuracy: dict[int | str, float],
    vocab: list[str] | None = None,
    save_dir: str | Path = "logs/plots",
    filename: str = "per_class_accuracy.png",
    top_n: int = 30,
    bottom_n: int = 30,
    return_fig: bool = False,
) -> object | None:
    """Plot a horizontal bar chart of per-class top-1 accuracy.

    Shows the ``top_n`` best-performing and ``bottom_n`` worst-performing
    classes side-by-side on a single figure to identify both easy and
    hard sign classes.

    Args:
        per_class_accuracy: Dict mapping class index (int) or label string
            to top-1 accuracy float.
        vocab: Optional ordered label list.  When ``per_class_accuracy`` uses
            integer keys, ``vocab[key]`` is used as the display label.
        save_dir: Directory to save the figure.
        filename: Output file name.
        top_n: Number of best classes to show.
        bottom_n: Number of worst classes to show.
        return_fig: Return the figure object instead of closing it.

    Returns:
        :class:`~pathlib.Path` to the saved PNG, or the figure if
        ``return_fig=True``.
    """
    import matplotlib.pyplot as plt

    # Normalise keys to label strings
    str_acc: dict[str, float] = {}
    for k, v in per_class_accuracy.items():
        label = (vocab[k] if vocab and k < len(vocab) else str(k)) if isinstance(k, int) else str(k)
        str_acc[label] = v

    sorted_acc = sorted(str_acc.items(), key=lambda x: x[1])
    worst = sorted_acc[:bottom_n]
    best = sorted_acc[-top_n:]

    fig, (ax_worst, ax_best) = plt.subplots(
        1,
        2,
        figsize=(16, max(6, max(top_n, bottom_n) * 0.28)),
        sharey=False,
    )
    fig.patch.set_facecolor("white")
    for ax in (ax_worst, ax_best):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

    # Worst classes
    worst_labels = [w[0] for w in worst]
    worst_vals = [w[1] for w in worst]
    ax_worst.barh(worst_labels, worst_vals, color="#D85A30", alpha=0.8)
    ax_worst.set_xlabel("Top-1 accuracy")
    ax_worst.set_title(f"Lowest {bottom_n} classes")
    ax_worst.set_xlim(0, 1)

    # Best classes
    best_labels = [b[0] for b in best]
    best_vals = [b[1] for b in best]
    ax_best.barh(best_labels, best_vals, color="#1D9E75", alpha=0.8)
    ax_best.set_xlabel("Top-1 accuracy")
    ax_best.set_title(f"Highest {top_n} classes")
    ax_best.set_xlim(0, 1)

    fig.suptitle("Per-class top-1 accuracy", fontsize=13, y=1.01)
    fig.tight_layout()

    out = _save(fig, save_dir, filename)
    if return_fig:
        return fig
    return out


# =============================================================================
# Embedding scatter (t-SNE)
# =============================================================================


def plot_embedding_scatter(
    embeddings: np.ndarray,
    labels: list[int],
    vocab: list[str],
    save_dir: str | Path = "logs/plots",
    filename: str = "embedding_scatter.png",
    max_samples: int = 1000,
    perplexity: float = 30.0,
    return_fig: bool = False,
) -> object | None:
    """Plot a 2-D t-SNE scatter of predicted video embeddings coloured by class.

    Reduces high-dimensional embeddings to 2D using t-SNE and renders a
    scatter plot where each point is one video clip, coloured by its true
    class label.  Well-separated clusters indicate the model has learned
    meaningful class-discriminative representations.

    Args:
        embeddings: Predicted embedding array ``(N, D)``, float32.
        labels: List of integer class indices, length ``N``.
        vocab: Class name list for legend labelling.
        save_dir: Directory to save the figure.
        filename: Output file name.
        max_samples: Maximum number of samples to include (subsampled
            randomly if ``N > max_samples``).
        perplexity: t-SNE perplexity parameter.  Typical values: 20–50.
        return_fig: Return the figure object instead of closing it.

    Returns:
        :class:`~pathlib.Path` to the saved PNG, or the figure if ``return_fig=True``.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE  # type: ignore[import]

    N = embeddings.shape[0]
    if max_samples < N:
        idx = np.random.choice(N, size=max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = [labels[i] for i in idx]
        log.info("t-SNE: subsampled %d → %d points", N, max_samples)

    log.info("Running t-SNE on %d samples (perplexity=%.1f) …", len(labels), perplexity)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    coords = tsne.fit_transform(embeddings)  # (N, 2)

    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    label_to_colour = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    fig, ax = _get_axes(figsize=(12, 9))
    ax.grid(alpha=0.2)

    for lbl in unique_labels:
        mask = [item == lbl for item in labels]
        pts = coords[[i for i, matched in enumerate(mask) if matched]]
        name = vocab[lbl] if lbl < len(vocab) else str(lbl)
        ax.scatter(pts[:, 0], pts[:, 1], c=[label_to_colour[lbl]], s=18, alpha=0.7, label=name)

    ax.set_title("t-SNE of predicted video embeddings (coloured by class)")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")

    # Only show legend if ≤ 30 classes (otherwise too cluttered)
    if len(unique_labels) <= 30:
        ax.legend(
            loc="upper right",
            fontsize=7,
            ncol=2,
            framealpha=0.7,
            markerscale=1.5,
        )

    out = _save(fig, save_dir, filename)
    if return_fig:
        return fig
    return out


# =============================================================================
# Cosine similarity distribution
# =============================================================================


def plot_cosine_similarity_distribution(
    pred_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    save_dir: str | Path = "logs/plots",
    filename: str = "cosine_sim_distribution.png",
    return_fig: bool = False,
) -> object | None:
    """Plot the distribution of cosine similarities between predicted and target.

    A distribution shifted towards 1.0 indicates the model is producing
    embeddings close to their Word2Vec targets.

    Args:
        pred_embeddings: Predicted embeddings ``(N, D)``, L2-normalised.
        target_embeddings: Target Word2Vec embeddings ``(N, D)``, L2-normalised.
        save_dir: Directory to save the figure.
        filename: Output file name.
        return_fig: Return the figure object instead of closing it.

    Returns:
        :class:`~pathlib.Path` to the saved PNG, or the figure if ``return_fig=True``.
    """

    cos_sims = (pred_embeddings * target_embeddings).sum(axis=1)

    fig, ax = _get_axes()
    ax.hist(cos_sims, bins=50, color="#534AB7", alpha=0.75, edgecolor="white")
    ax.axvline(
        cos_sims.mean(),
        color="#D85A30",
        linewidth=2,
        label=f"Mean: {cos_sims.mean():.3f}",
    )
    ax.set_xlabel("Cosine similarity (predicted vs. Word2Vec target)")
    ax.set_ylabel("Number of samples")
    ax.set_title("Cosine similarity distribution")
    ax.set_xlim(-1, 1)
    ax.legend()

    out = _save(fig, save_dir, filename)
    if return_fig:
        return fig
    return out


# =============================================================================
# Training summary (all plots in one call)
# =============================================================================


def plot_training_summary(
    train_losses: list[float],
    val_losses: list[float],
    top1_history: list[float],
    top5_history: list[float],
    throughput_history: list[float],
    save_dir: str | Path = "logs/plots",
) -> None:
    """Generate and save all standard training diagnostic plots.

    Convenience wrapper that calls all individual plot functions in sequence.
    Saves five files to ``save_dir``:
    ``loss_curves.png``, ``accuracy_curves.png``, ``throughput.png``.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses.
        top1_history: Per-epoch validation top-1 accuracy.
        top5_history: Per-epoch validation top-5 accuracy.
        throughput_history: Per-epoch training throughput (clips/s).
        save_dir: Directory to write all plots.
    """
    plot_loss_curves(train_losses, val_losses, save_dir=save_dir)
    plot_accuracy_curves(top1_history, top5_history, save_dir=save_dir)
    if throughput_history:
        plot_throughput(throughput_history, save_dir=save_dir)
    log.info("All training summary plots saved to %s", save_dir)
