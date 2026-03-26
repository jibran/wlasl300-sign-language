"""Inference script for the WLASL300 sign language classifier.

Loads a trained :class:`~models.sign_model_classifier.SignModelClassifier`
checkpoint and runs top-k classification on one or more video clips.  Unlike
the embedding-model inference (``inference/inference.py``), no Word2Vec
class matrix is required — predictions are produced by a direct argmax over
the 300-class logit vector.

**Single-video mode**::

    uv run python inference/inference_classifier.py \\
        --checkpoint trained_models/classifier/best/checkpoint.pt \\
        --video WLASL300/0/00412.mp4 \\
        --top_k 5

**Batch-directory mode**::

    uv run python inference/inference_classifier.py \\
        --checkpoint trained_models/classifier/best/checkpoint.pt \\
        --video_dir WLASL300/ \\
        --output results_classifier.json

Output (``--top_k 5``)::

    {
      "video": "WLASL300/0/00412.mp4",
      "predictions": [
        {"rank": 1, "label": "book",    "score": 0.942},
        {"rank": 2, "label": "read",    "score": 0.031},
        ...
      ],
      "inference_time_ms": 43.1
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from config.base_config import Config
from dataset.data.wlasl_dataset import preprocess_live_frames
from inference.inference import _decode_frames
from models.sign_model_classifier import SignModelClassifier
from models.sign_model_deep import SignModelDeep
from models.sign_model_linear import SignModelLinear
from models.sign_model_temporal import SignModelTemporal

log = logging.getLogger(__name__)

AnyClassifier = SignModelClassifier | SignModelLinear | SignModelDeep | SignModelTemporal


# =============================================================================
# Device
# =============================================================================


def _resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Args:
        device_str: ``"cuda"``, ``"cpu"``, or ``"auto"``.

    Returns:
        Resolved :class:`torch.device`.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# =============================================================================
# Single-video prediction
# =============================================================================


def predict_single(
    video_path: str,
    model: AnyClassifier,
    vocab: list[str],
    cfg: Config,
    device: torch.device,
    top_k: int = 5,
) -> dict:
    """Run classifier inference on a single video file.

    Decodes ``video_path``, applies the val-mode preprocessing pipeline, and
    returns the top-k class predictions with softmax probability scores.

    Args:
        video_path: Path to the input ``.mp4`` file.
        model: Loaded :class:`~models.sign_model_classifier.SignModelClassifier`
            in eval mode.
        vocab: Ordered class label list (``vocab[class_idx] == label``).
        cfg: Project config.
        device: Inference device.
        top_k: Number of top predictions to return.

    Returns:
        Dict with ``"video"``, ``"predictions"``, and ``"inference_time_ms"``
        keys.  ``"predictions"`` is a list of ``{"rank", "label", "score"}``
        dicts ordered by score.  Returns an ``"error"`` key if decoding fails.
    """
    frames = _decode_frames(
        video_path=video_path,
        num_frames=cfg.dataset.num_frames,
        loop_short=True,
    )
    if frames is None:
        log.error("Failed to decode: %s", video_path)
        return {"video": video_path, "error": "decode failed"}

    t0 = time.perf_counter()
    video_tensor = preprocess_live_frames(frames, cfg, device=device)

    with torch.no_grad():
        predictions = model.predict_topk(video_tensor, vocab=vocab, k=top_k)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "video": video_path,
        "predictions": predictions[0],
        "inference_time_ms": round(elapsed_ms, 1),
    }


# =============================================================================
# Batch-directory prediction
# =============================================================================


def predict_directory(
    video_dir: Path,
    model: AnyClassifier,
    vocab: list[str],
    cfg: Config,
    device: torch.device,
    top_k: int = 5,
) -> list[dict]:
    """Run classifier inference on all ``.mp4`` files under ``video_dir``.

    Args:
        video_dir: Root directory to search for ``.mp4`` files recursively.
        model: Loaded classifier in eval mode.
        vocab: Ordered class label list.
        cfg: Project config.
        device: Inference device.
        top_k: Number of top predictions per clip.

    Returns:
        List of result dicts (one per video), same format as
        :func:`predict_single`.
    """
    mp4_files = sorted(video_dir.rglob("*.mp4"))
    log.info("Running classifier inference on %d files in %s …", len(mp4_files), video_dir)

    results = []
    for i, mp4 in enumerate(mp4_files, 1):
        result = predict_single(
            video_path=str(mp4),
            model=model,
            vocab=vocab,
            cfg=cfg,
            device=device,
            top_k=top_k,
        )
        results.append(result)
        if i % 50 == 0:
            log.info("  %d / %d done", i, len(mp4_files))

    return results


# =============================================================================
# Output formatting
# =============================================================================


def format_result(result: dict, verbose: bool = False) -> str:
    """Format a single prediction result as a human-readable string.

    Args:
        result: Result dict from :func:`predict_single`.
        verbose: If ``True``, include all top-k predictions.

    Returns:
        Formatted string.
    """
    if "error" in result:
        return f"ERROR  {result['video']}  —  {result['error']}"

    lines = [f"Video : {result['video']}"]
    preds = result.get("predictions", [])
    n_show = len(preds) if verbose else 1
    for p in preds[:n_show]:
        lines.append(f"  #{p['rank']:>2}  {p['label']:<20}  {p['score']:.4f}")
    if not verbose and preds:
        lines.append(f"       (top-{len(preds)} available — use --verbose)")
    lines.append(f"Time  : {result.get('inference_time_ms', 0):.1f} ms")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    p = argparse.ArgumentParser(
        description="Run sign language classifier inference on video clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to a single .mp4 file.")
    group.add_argument("--video_dir", type=str, help="Directory of .mp4 files.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (.pt). Defaults based on --model flag.",
    )
    p.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="classifier",
        choices=["classifier", "linear", "deep", "temporal"],
        help="Model head type used during training.",
    )
    p.add_argument("--top_k", type=int, default=5, help="Number of top predictions.")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write results as JSON to this path.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Inference device.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print all top-k predictions.",
    )
    return p.parse_args()


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    """Entry point for classifier inference."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    cfg = Config.from_yaml(args.config)
    device = _resolve_device(args.device)
    log.info("Inference device: %s", device)

    if args.model == "linear":
        ModelClass = SignModelLinear
    elif args.model == "deep":
        ModelClass = SignModelDeep
    elif args.model == "temporal":
        ModelClass = SignModelTemporal
    else:
        ModelClass = SignModelClassifier
    default_ckpt = f"trained_models/{args.model}/best/checkpoint.pt"
    ckpt_path = Path(args.checkpoint or default_ckpt)
    if not ckpt_path.exists():
        log.error(
            "Checkpoint not found: %s\n"
            "Train first: uv run python train/train_classifier.py --model %s",
            args.model,
            ckpt_path,
        )
        raise SystemExit(1)

    log.info("Loading checkpoint: %s", ckpt_path)
    model, epoch, metrics = ModelClass.load_checkpoint(str(ckpt_path), cfg, device=str(device))
    model.eval()
    log.info(
        "Checkpoint loaded — epoch=%d  val_top1=%.4f",
        epoch,
        metrics.get("top1", 0.0),
    )

    # Load vocab
    vocab: list[str] = json.loads(Path(cfg.paths.vocab_file).read_text(encoding="utf-8"))
    log.info("Vocab: %d classes", len(vocab))

    # Run inference
    if args.video:
        result = predict_single(
            video_path=args.video,
            model=model,
            vocab=vocab,
            cfg=cfg,
            device=device,
            top_k=args.top_k,
        )
        print(format_result(result, verbose=args.verbose or True))

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            log.info("Result written → %s", args.output)

    else:
        results = predict_directory(
            video_dir=Path(args.video_dir),
            model=model,
            vocab=vocab,
            cfg=cfg,
            device=device,
            top_k=args.top_k,
        )

        if args.verbose:
            for r in results:
                print(format_result(r, verbose=True))
                print()

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            log.info("Results written → %s  (%d videos)", args.output, len(results))
        else:
            for r in results:
                print(format_result(r))


if __name__ == "__main__":
    main()
