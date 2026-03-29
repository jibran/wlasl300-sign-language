"""UMAP / t-SNE visualisation of backbone feature embeddings.

Extracts the backbone's feature vector for every clip in the val or test
split, reduces to 2-D with UMAP or t-SNE, and saves an interactive HTML
scatter plot (Plotly) and a static PNG (matplotlib).

Each point represents one video clip, coloured by its class label.
Well-separated, tight clusters indicate the backbone has learned
discriminative sign-specific representations independently of the head.

Works with every model type (classifier, deep, linear, temporal) because
it taps ``I3DBackbone.forward()`` directly, bypassing the neck and head.

Usage::

    # Val split, UMAP, temporal model
    uv run python inference/visualise_embeddings.py \\
        --checkpoint trained_models/temporal/best/checkpoint.pt \\
        --model temporal \\
        --split val \\
        --method umap \\
        --out_dir logs/temporal/embeddings

    # Test split, t-SNE, classifier model
    uv run python inference/visualise_embeddings.py \\
        --checkpoint trained_models/classifier/best/checkpoint.pt \\
        --model classifier \\
        --split test \\
        --method tsne \\
        --n_classes 50

Output files::

    logs/temporal/embeddings/
        umap_val.html      # interactive Plotly scatter (hover = label)
        umap_val.png       # static matplotlib scatter
        umap_val_raw.npz   # raw features + labels for further analysis

"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.base_config import Config
from dataset.data.wlasl_dataset import WLASL300Dataset
from models.sign_model_classifier import SignModelClassifier
from models.sign_model_deep import SignModelDeep
from models.sign_model_linear import SignModelLinear
from models.sign_model_temporal import SignModelTemporal

log = logging.getLogger(__name__)

AnyModel = SignModelClassifier | SignModelDeep | SignModelLinear | SignModelTemporal

_MODEL_CLASSES = {
    "classifier": SignModelClassifier,
    "deep": SignModelDeep,
    "linear": SignModelLinear,
    "temporal": SignModelTemporal,
}


# =============================================================================
# Feature extraction
# =============================================================================


@torch.no_grad()
def extract_features(
    model: AnyModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract backbone feature vectors for every clip in the loader.

    Runs only the backbone (``I3DBackbone.forward()``) — the neck and head
    are intentionally skipped so the visualisation shows what the backbone
    itself has learned, independent of any downstream component.

    Args:
        model: Any loaded classifier model.
        loader: DataLoader yielding ``(video, embedding, label_idx)`` batches.
        device: Compute device.

    Returns:
        Tuple of:
        - ``features``: ``(N, D)`` float32 array of backbone feature vectors.
        - ``labels``: ``(N,)`` int32 array of class indices.
        - ``label_names``: list of string label names aligned to ``labels``.
    """
    model.eval()
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    total = len(loader.dataset)
    log.info("Extracting features from %d clips …", total)
    t0 = time.perf_counter()

    for batch_idx, (videos, _emb, label_idx) in enumerate(loader):
        videos = videos.to(device, non_blocking=True)

        # Always use backbone.forward() — never the neck or head.
        # This gives a (B, D) flat feature vector regardless of model type.
        feats = model.backbone(videos)           # (B, 2048)

        all_features.append(feats.cpu().float().numpy())
        all_labels.append(label_idx.numpy().astype(np.int32))

        if (batch_idx + 1) % 20 == 0:
            done = (batch_idx + 1) * loader.batch_size
            log.info("  %d / %d clips  (%.0f clips/s)",
                     min(done, total), total,
                     min(done, total) / max(time.perf_counter() - t0, 1e-6))

    features = np.concatenate(all_features, axis=0)
    labels   = np.concatenate(all_labels,   axis=0)
    elapsed  = time.perf_counter() - t0

    log.info(
        "Extraction complete — %d clips  shape=%s  time=%.1fs",
        len(features), features.shape, elapsed,
    )
    return features, labels, []


# =============================================================================
# Dimensionality reduction
# =============================================================================


def reduce_umap(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce features to 2-D with UMAP.

    Args:
        features: ``(N, D)`` feature matrix.
        n_neighbors: UMAP n_neighbors — controls local vs global structure.
            Lower = more local detail, higher = more global topology.
        min_dist: Minimum distance between points in the embedding.
        random_state: Random seed for reproducibility.

    Returns:
        ``(N, 2)`` 2-D embedding array.
    """
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "UMAP is not installed. Install it with:\n"
            "  uv add umap-learn\n"
            "or:\n"
            "  pip install umap-learn"
        ) from exc

    log.info(
        "Running UMAP  n_neighbors=%d  min_dist=%.2f  n_samples=%d …",
        n_neighbors, min_dist, len(features),
    )
    t0 = time.perf_counter()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        low_memory=False,
    )
    embedding = reducer.fit_transform(features)
    log.info("UMAP done in %.1fs", time.perf_counter() - t0)
    return embedding.astype(np.float32)


def reduce_tsne(
    features: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    pca_components: int = 50,
) -> np.ndarray:
    """Reduce features to 2-D with t-SNE.

    First applies PCA to reduce to ``pca_components`` dimensions (speeds up
    t-SNE significantly and removes noise), then runs t-SNE to 2-D.

    Args:
        features: ``(N, D)`` feature matrix.
        perplexity: t-SNE perplexity — roughly the number of near neighbours
            each point considers. Typical range: 5–50.
        n_iter: Number of t-SNE optimisation iterations.
        random_state: Random seed.
        pca_components: PCA pre-reduction dimension. Set to 0 to skip PCA.

    Returns:
        ``(N, 2)`` 2-D embedding array.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if pca_components > 0 and features.shape[1] > pca_components:
        log.info("PCA pre-reduction: %d → %d dims …", features.shape[1], pca_components)
        t0 = time.perf_counter()
        pca = PCA(n_components=pca_components, random_state=random_state)
        features = pca.fit_transform(features)
        explained = pca.explained_variance_ratio_.sum() * 100
        log.info(
            "PCA done in %.1fs  (%.1f%% variance retained)",
            time.perf_counter() - t0, explained,
        )

    log.info(
        "Running t-SNE  perplexity=%.0f  n_iter=%d  n_samples=%d …",
        perplexity, n_iter, len(features),
    )
    t0 = time.perf_counter()
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1,
    )
    embedding = tsne.fit_transform(features)
    log.info("t-SNE done in %.1fs", time.perf_counter() - t0)
    return embedding.astype(np.float32)


# =============================================================================
# Plotting
# =============================================================================


def _class_colours(n: int) -> list[str]:
    """Generate ``n`` perceptually distinct hex colours using HSL cycling."""
    import colorsys
    colours = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.72)
        colours.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colours


def save_plotly_html(
    embedding: np.ndarray,
    labels: np.ndarray,
    vocab: list[str],
    out_path: Path,
    method: str,
    split: str,
    model_type: str,
    n_classes_shown: int,
) -> None:
    """Save an interactive Plotly scatter plot to an HTML file.

    Args:
        embedding: ``(N, 2)`` 2-D coordinates.
        labels: ``(N,)`` integer class indices.
        vocab: Full class label list.
        out_path: Destination ``.html`` file path.
        method: ``"umap"`` or ``"tsne"`` — used in axis labels.
        split: ``"val"`` or ``"test"``.
        model_type: Model type string for the plot title.
        n_classes_shown: Number of classes included in the plot.
    """
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Plotly is not installed. Install it with:\n"
            "  uv add plotly\nor:\n"
            "  pip install plotly"
        ) from exc

    unique_labels = np.unique(labels)
    colours = _class_colours(len(unique_labels))
    label_to_colour = {lbl: colours[i] for i, lbl in enumerate(unique_labels)}

    traces = []
    for lbl in unique_labels:
        mask = labels == lbl
        name = vocab[lbl] if lbl < len(vocab) else str(lbl)
        traces.append(go.Scatter(
            x=embedding[mask, 0],
            y=embedding[mask, 1],
            mode="markers",
            marker=dict(size=5, color=label_to_colour[lbl], opacity=0.75),
            name=name,
            text=[name] * mask.sum(),
            hovertemplate="<b>%{text}</b><br>x=%{x:.2f}  y=%{y:.2f}<extra></extra>",
        ))

    method_upper = method.upper()
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(
            text=f"{model_type} backbone — {method_upper} of {split} features "
                 f"({n_classes_shown} classes)",
            font=dict(size=14),
        ),
        xaxis_title=f"{method_upper} dim 1",
        yaxis_title=f"{method_upper} dim 2",
        showlegend=True,
        legend=dict(
            itemsizing="constant",
            font=dict(size=9),
            tracegroupgap=2,
        ),
        width=1000,
        height=750,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.write_html(str(out_path))
    log.info("Interactive plot → %s", out_path)


def save_matplotlib_png(
    embedding: np.ndarray,
    labels: np.ndarray,
    vocab: list[str],
    out_path: Path,
    method: str,
    split: str,
    model_type: str,
    n_classes_shown: int,
) -> None:
    """Save a static matplotlib scatter plot to a PNG file.

    Args:
        embedding: ``(N, 2)`` 2-D coordinates.
        labels: ``(N,)`` integer class indices.
        vocab: Full class label list.
        out_path: Destination ``.png`` file path.
        method: ``"umap"`` or ``"tsne"``.
        split: ``"val"`` or ``"test"``.
        model_type: Model type string for the title.
        n_classes_shown: Number of classes in the plot.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    unique_labels = np.unique(labels)
    cmap = cm.get_cmap("tab20", len(unique_labels))

    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = vocab[lbl] if lbl < len(vocab) else str(lbl)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=12,
            alpha=0.7,
            color=cmap(i),
            label=name,
            linewidths=0,
        )

    method_upper = method.upper()
    ax.set_title(
        f"{model_type} backbone — {method_upper} of {split} features "
        f"({n_classes_shown} classes)",
        fontsize=12,
    )
    ax.set_xlabel(f"{method_upper} dim 1")
    ax.set_ylabel(f"{method_upper} dim 2")
    ax.set_facecolor("#f8f8f8")

    if n_classes_shown <= 30:
        ax.legend(
            loc="best",
            fontsize=6,
            ncol=2,
            framealpha=0.6,
            markerscale=1.5,
        )

    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    log.info("Static plot → %s", out_path)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="UMAP / t-SNE visualisation of backbone feature embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained model checkpoint (.pt).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="temporal",
        choices=list(_MODEL_CLASSES.keys()),
        help="Model type matching the checkpoint.",
    )
    p.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to visualise.",
    )
    p.add_argument(
        "--method",
        type=str,
        default="umap",
        choices=["umap", "tsne", "both"],
        help="Dimensionality reduction method.",
    )
    p.add_argument(
        "--n_classes",
        type=int,
        default=0,
        help="Visualise only the N most frequent classes. 0 = all 300.",
    )
    p.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (ignored for t-SNE).",
    )
    p.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (ignored for t-SNE).",
    )
    p.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (ignored for UMAP).",
    )
    p.add_argument(
        "--n_iter",
        type=int,
        default=1000,
        help="t-SNE iterations (ignored for UMAP).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for feature extraction.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Inference device.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to logs/<model>/embeddings.",
    )
    p.add_argument(
        "--no_html",
        action="store_true",
        default=False,
        help="Skip the interactive Plotly HTML output.",
    )
    p.add_argument(
        "--no_png",
        action="store_true",
        default=False,
        help="Skip the static matplotlib PNG output.",
    )
    return p.parse_args()


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    """Entry point for the embedding visualisation script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    cfg = Config.from_yaml(args.config)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    log.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────────
    ModelClass = _MODEL_CLASSES[args.model]
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log.info("Loading checkpoint: %s", ckpt_path)
    model, epoch, metrics = ModelClass.load_checkpoint(
        str(ckpt_path), cfg, device=device_str
    )
    model.eval()
    model.to(device)
    log.info("Checkpoint loaded — epoch=%d  val_top1=%.4f", epoch, metrics.get("top1", 0))

    # ── Load vocab ────────────────────────────────────────────────────────────
    vocab: list[str] = json.loads(
        Path(cfg.paths.vocab_file).read_text(encoding="utf-8")
    )

    # ── Build dataset and loader ──────────────────────────────────────────────
    dataset = WLASL300Dataset(
        split=args.split,
        annotations_file=cfg.paths.annotations_file,
        embeddings_file=cfg.paths.embeddings_file,
        vocab_file=cfg.paths.vocab_file,
        cfg=cfg,
        cache_dir=cfg.paths.processed_dir,
    )
    log.info("Dataset: %d clips  split=%s", len(dataset), args.split)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    # ── Extract features ──────────────────────────────────────────────────────
    features, labels, _ = extract_features(model, loader, device)

    # ── Optionally filter to top-N classes ───────────────────────────────────
    n_classes_shown = len(np.unique(labels))
    if args.n_classes > 0 and args.n_classes < n_classes_shown:
        from collections import Counter
        counts = Counter(labels.tolist())
        top_classes = {cls for cls, _ in counts.most_common(args.n_classes)}
        mask = np.array([l in top_classes for l in labels])
        features = features[mask]
        labels   = labels[mask]
        n_classes_shown = args.n_classes
        log.info(
            "Filtered to top %d classes — %d clips remaining",
            n_classes_shown, len(features),
        )

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(f"logs/{args.model}/embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Save raw features ─────────────────────────────────────────────────────
    raw_path = out_dir / f"{args.method}_{args.split}_raw.npz"
    np.savez_compressed(str(raw_path), features=features, labels=labels)
    log.info("Raw features → %s  (shape=%s)", raw_path, features.shape)

    # ── Reduce and plot ───────────────────────────────────────────────────────
    methods = ["umap", "tsne"] if args.method == "both" else [args.method]

    for method in methods:
        if method == "umap":
            embedding = reduce_umap(
                features,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
            )
        else:
            embedding = reduce_tsne(
                features,
                perplexity=args.perplexity,
                n_iter=args.n_iter,
            )

        stem = f"{method}_{args.split}"

        if not args.no_html:
            save_plotly_html(
                embedding, labels, vocab,
                out_dir / f"{stem}.html",
                method, args.split, args.model, n_classes_shown,
            )

        if not args.no_png:
            save_matplotlib_png(
                embedding, labels, vocab,
                out_dir / f"{stem}.png",
                method, args.split, args.model, n_classes_shown,
            )

        emb_path = out_dir / f"{stem}_coords.npy"
        np.save(str(emb_path), embedding)
        log.info("2-D coordinates → %s", emb_path)

    log.info("Done. Outputs in: %s", out_dir)


if __name__ == "__main__":
    main()
