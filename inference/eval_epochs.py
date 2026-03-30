"""Evaluate saved per-epoch checkpoints for validation and test accuracy.

Loads every ``epoch_NNN.pt`` checkpoint from a directory, runs the full val
and (optionally) test evaluation loop on each, and writes a CSV summary plus
an interactive HTML chart.

Designed for runs where ``--skip_val --skip_test --save_all_epochs`` was used
during training, so validation and test metrics need to be computed offline.

Usage::

    # Val only (faster — skip test to save time)
    uv run python inference/eval_epochs.py \\
        --epochs_dir trained_models/classifier/epochs \\
        --model classifier \\
        --skip_test

    # Val + test for every epoch
    uv run python inference/eval_epochs.py \\
        --epochs_dir trained_models/classifier/epochs \\
        --model classifier

    # Specific epoch range
    uv run python inference/eval_epochs.py \\
        --epochs_dir trained_models/classifier/epochs \\
        --model classifier \\
        --start_epoch 10 \\
        --end_epoch 30

Output::

    logs/classifier/epoch_eval/
        results.csv          ← epoch, val_top1, val_top5, val_loss,
                                       test_top1, test_top5, test_loss
        curves.html          ← interactive Plotly chart
        best_val.json        ← best val checkpoint metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from config.base_config import Config
from dataset.data.wlasl_dataset import build_dataloaders
from models.sign_model_classifier import SignModelClassifier
from models.sign_model_deep import SignModelDeep
from models.sign_model_linear import SignModelLinear
from models.sign_model_temporal import SignModelTemporal
from train.train_classifier import evaluate, topk_accuracy

log = logging.getLogger(__name__)

_MODEL_CLASSES = {
    "classifier": SignModelClassifier,
    "deep":       SignModelDeep,
    "linear":     SignModelLinear,
    "temporal":   SignModelTemporal,
}


# =============================================================================
# Checkpoint discovery
# =============================================================================


def discover_checkpoints(
    epochs_dir: Path,
    start_epoch: int,
    end_epoch: int,
) -> list[tuple[int, Path]]:
    """Discover and sort epoch checkpoint files in ``epochs_dir``.

    Accepts filenames matching ``epoch_NNN.pt`` (1-based epoch number).

    Args:
        epochs_dir: Directory containing ``epoch_NNN.pt`` files.
        start_epoch: First epoch to evaluate (1-based, inclusive).
        end_epoch: Last epoch to evaluate (1-based, inclusive). 0 = no limit.

    Returns:
        Sorted list of ``(epoch_number, path)`` tuples.

    Raises:
        FileNotFoundError: If ``epochs_dir`` does not exist.
        RuntimeError: If no matching checkpoints are found.
    """
    if not epochs_dir.exists():
        raise FileNotFoundError(f"Epochs directory not found: {epochs_dir}")

    pattern = re.compile(r"^epoch_(\d+)\.pt$")
    found: list[tuple[int, Path]] = []

    for f in sorted(epochs_dir.iterdir()):
        m = pattern.match(f.name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep < start_epoch:
            continue
        if end_epoch > 0 and ep > end_epoch:
            continue
        found.append((ep, f))

    found.sort(key=lambda x: x[0])

    if not found:
        raise RuntimeError(
            f"No epoch_NNN.pt checkpoints found in {epochs_dir} "
            f"(range {start_epoch}–{end_epoch or '∞'})"
        )

    log.info(
        "Found %d checkpoints  (epoch %d → %d)",
        len(found), found[0][0], found[-1][0],
    )
    return found


# =============================================================================
# CSV writer
# =============================================================================


def _write_csv(rows: list[dict], out_path: Path) -> None:
    """Write results to a CSV file.

    Args:
        rows: List of result dicts (one per epoch).
        out_path: Destination CSV path.
    """
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    lines = [",".join(fieldnames)]
    for row in rows:
        lines.append(",".join(str(row[k]) for k in fieldnames))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Results CSV → %s", out_path)


# =============================================================================
# Plotly chart
# =============================================================================


def _write_plotly_chart(rows: list[dict], out_path: Path, model_type: str) -> None:
    """Write an interactive Plotly HTML chart.

    Args:
        rows: List of result dicts (one per epoch).
        out_path: Destination HTML path.
        model_type: Model type string for the chart title.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning(
            "Plotly not installed — skipping HTML chart. "
            "Install with: pip install plotly"
        )
        return

    epochs   = [r["epoch"] for r in rows]
    val_t1   = [r["val_top1"] * 100  for r in rows]
    val_t5   = [r["val_top5"] * 100  for r in rows]
    val_loss = [r["val_loss"]         for r in rows]

    has_test = any(r.get("test_top1") is not None for r in rows)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top-1 and top-5 accuracy (%)", "Val loss"),
    )

    fig.add_trace(go.Scatter(
        x=epochs, y=val_t1, name="val top-1",
        line=dict(color="#534AB7", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epochs, y=val_t5, name="val top-5",
        line=dict(color="#534AB7", width=1.5, dash="dot"),
    ), row=1, col=1)

    if has_test:
        test_t1 = [r.get("test_top1", None) for r in rows]
        test_t5 = [r.get("test_top5", None) for r in rows]
        fig.add_trace(go.Scatter(
            x=epochs, y=[v * 100 if v is not None else None for v in test_t1],
            name="test top-1",
            line=dict(color="#1D9E75", width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=epochs, y=[v * 100 if v is not None else None for v in test_t5],
            name="test top-5",
            line=dict(color="#1D9E75", width=1.5, dash="dot"),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=val_loss, name="val loss",
        line=dict(color="#D85A30", width=2),
        showlegend=True,
    ), row=1, col=2)

    best_ep = max(rows, key=lambda r: r["val_top1"])
    fig.add_vline(
        x=best_ep["epoch"], line_dash="dash", line_color="gray",
        annotation_text=f"best val @ epoch {best_ep['epoch']}",
        annotation_font_size=10,
        row=1, col=1,
    )

    fig.update_layout(
        title=dict(
            text=f"{model_type} — per-epoch validation"
                 + (" + test" if has_test else ""),
            font=dict(size=14),
        ),
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Loss",          row=1, col=2)

    fig.write_html(str(out_path))
    log.info("Interactive chart → %s", out_path)


# =============================================================================
# Main evaluation loop
# =============================================================================


def run_evaluation(
    epochs_dir: Path,
    model_type: str,
    cfg: Config,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    device: torch.device,
    out_dir: Path,
    resume_csv: Path | None,
) -> None:
    """Evaluate every checkpoint and write results.

    Args:
        epochs_dir: Directory with ``epoch_NNN.pt`` checkpoints.
        model_type: Model type string.
        cfg: Project config.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader or ``None`` if ``--skip_test`` was given.
        device: Compute device.
        out_dir: Output directory for CSV, chart, and best checkpoint JSON.
        resume_csv: Existing CSV to resume from (skips already-computed epochs).
    """
    ModelClass = _MODEL_CLASSES[model_type]

    # Load already-computed epochs from a previous partial run
    already_done: dict[int, dict] = {}
    if resume_csv and resume_csv.exists():
        import csv
        with open(resume_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ep = int(row["epoch"])
                already_done[ep] = {
                    "epoch":     ep,
                    "val_top1":  float(row["val_top1"]),
                    "val_top5":  float(row["val_top5"]),
                    "val_loss":  float(row["val_loss"]),
                    "test_top1": float(row["test_top1"]) if row.get("test_top1") else None,
                    "test_top5": float(row["test_top5"]) if row.get("test_top5") else None,
                    "test_loss": float(row["test_loss"]) if row.get("test_loss") else None,
                }
        log.info("Resuming — %d epochs already evaluated", len(already_done))

    checkpoints = discover_checkpoints(
        epochs_dir,
        start_epoch=args_global.start_epoch,
        end_epoch=args_global.end_epoch,
    )

    rows: list[dict] = list(already_done.values())
    total = len(checkpoints)

    for idx, (epoch_num, ckpt_path) in enumerate(checkpoints, 1):
        if epoch_num in already_done:
            log.info("Skipping epoch %d (already in CSV)", epoch_num)
            continue

        log.info(
            "─── Epoch %d / %d  (%s) ───",
            epoch_num, checkpoints[-1][0], ckpt_path.name,
        )
        t0 = time.perf_counter()

        # Load only the model weights — no optimiser/scheduler state needed
        model, _, train_metrics = ModelClass.load_checkpoint(
            str(ckpt_path), cfg, device=str(device)
        )
        model.eval()

        # Val
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            cfg=cfg,
            epoch=epoch_num - 1,  # 0-based inside evaluate()
            split="val",
            device=device,
        )

        # Test (optional)
        test_metrics: dict | None = None
        if test_loader is not None:
            test_metrics = evaluate(
                model=model,
                loader=test_loader,
                cfg=cfg,
                epoch=epoch_num - 1,
                split="test",
                device=device,
            )

        elapsed = time.perf_counter() - t0
        row: dict = {
            "epoch":     epoch_num,
            "val_top1":  round(val_metrics["top1"], 6),
            "val_top5":  round(val_metrics["top5"], 6),
            "val_loss":  round(val_metrics["loss"], 6),
            "test_top1": round(test_metrics["top1"], 6) if test_metrics else None,
            "test_top5": round(test_metrics["top5"], 6) if test_metrics else None,
            "test_loss": round(test_metrics["loss"], 6) if test_metrics else None,
        }
        rows.append(row)

        log.info(
            "  val  top1=%.3f  top5=%.3f  loss=%.4f%s  (%.1fs)",
            val_metrics["top1"],
            val_metrics["top5"],
            val_metrics["loss"],
            f"  |  test top1={test_metrics['top1']:.3f}  top5={test_metrics['top5']:.3f}"
            if test_metrics else "",
            elapsed,
        )

        # Write CSV incrementally so progress is saved after each epoch
        sorted_rows = sorted(rows, key=lambda r: r["epoch"])
        _write_csv(sorted_rows, out_dir / "results.csv")

        # Free GPU memory before loading next checkpoint
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Final outputs ─────────────────────────────────────────────────────────
    sorted_rows = sorted(rows, key=lambda r: r["epoch"])

    _write_csv(sorted_rows, out_dir / "results.csv")
    _write_plotly_chart(sorted_rows, out_dir / "curves.html", model_type)

    # Best val checkpoint summary
    best = max(sorted_rows, key=lambda r: r["val_top1"])
    best_path = out_dir / "best_val.json"
    best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    log.info(
        "Best val top-1: %.4f  at epoch %d  → %s",
        best["val_top1"], best["epoch"], best_path,
    )

    # Pretty print final table
    print("\n" + "─" * 72)
    header = f"{'Epoch':>6}  {'Val top-1':>10}  {'Val top-5':>10}  {'Val loss':>9}"
    if any(r.get("test_top1") is not None for r in sorted_rows):
        header += f"  {'Test top-1':>10}  {'Test top-5':>10}"
    print(header)
    print("─" * 72)
    for r in sorted_rows:
        marker = " ←" if r["epoch"] == best["epoch"] else ""
        line = (
            f"{r['epoch']:>6}  "
            f"{r['val_top1']*100:>9.2f}%  "
            f"{r['val_top5']*100:>9.2f}%  "
            f"{r['val_loss']:>9.4f}"
        )
        if r.get("test_top1") is not None:
            line += (
                f"  {r['test_top1']*100:>9.2f}%"
                f"  {r['test_top5']*100:>9.2f}%"
            )
        print(line + marker)
    print("─" * 72)
    print(
        f"Best val top-1: {best['val_top1']*100:.2f}%  "
        f"(top-5: {best['val_top5']*100:.2f}%)  "
        f"at epoch {best['epoch']}"
    )


# =============================================================================
# CLI
# =============================================================================

# Global reference so discover_checkpoints can read start/end epoch
args_global: argparse.Namespace


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Evaluate saved per-epoch checkpoints for val / test accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--epochs_dir",
        type=str,
        required=True,
        help="Directory containing epoch_NNN.pt checkpoints.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="classifier",
        choices=list(_MODEL_CLASSES.keys()),
        help="Model type matching the checkpoints.",
    )
    p.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml.",
    )
    p.add_argument(
        "--start_epoch",
        type=int,
        default=1,
        help="First epoch to evaluate (1-based, inclusive).",
    )
    p.add_argument(
        "--end_epoch",
        type=int,
        default=0,
        help="Last epoch to evaluate (1-based, inclusive). 0 = all.",
    )
    p.add_argument(
        "--skip_test",
        action="store_true",
        default=False,
        help="Skip test evaluation (only compute val metrics).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override config val/test batch size.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Evaluation device.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to logs/<model>/epoch_eval.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from an existing results.csv, skipping already-evaluated epochs.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for epoch evaluation."""
    global args_global

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    args_global = args

    cfg = Config.from_yaml(args.config)

    # Device
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    log.info("Device: %s", device)

    # Batch size override
    if args.batch_size is not None:
        import dataclasses
        new_training = dataclasses.replace(cfg.training, batch_size=args.batch_size)
        cfg = dataclasses.replace(cfg, training=new_training)

    # Build val + test loaders only (no train loader needed)
    log.info("Building val / test dataloaders …")
    _, val_loader, test_loader = build_dataloaders(
        cfg,
        skip_test=args.skip_test,
    )
    log.info(
        "Val: %d batches  |  Test: %s",
        len(val_loader),
        f"{len(test_loader)} batches" if test_loader else "skipped",
    )

    # Output directory
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(f"logs/{args.model}/epoch_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    resume_csv = (out_dir / "results.csv") if args.resume else None

    run_evaluation(
        epochs_dir=Path(args.epochs_dir),
        model_type=args.model,
        cfg=cfg,
        val_loader=val_loader,
        test_loader=test_loader if not args.skip_test else None,
        device=device,
        out_dir=out_dir,
        resume_csv=resume_csv,
    )


if __name__ == "__main__":
    main()
