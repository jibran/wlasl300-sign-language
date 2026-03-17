"""Inference script for WLASL300 sign language recognition.

Loads a trained :class:`~models.sign_model.SignModel` checkpoint and runs
nearest-neighbour prediction on one or more video clips.  Supports two modes:

**Single-video mode**::

    uv run python inference/inference.py \\
        --checkpoint trained_models/best/checkpoint.pt \\
        --video path/to/video.mp4 \\
        --top_k 5

**Batch-directory mode** (all ``.mp4`` files under ``--video_dir``)::

    uv run python inference/inference.py \\
        --checkpoint trained_models/best/checkpoint.pt \\
        --video_dir path/to/videos/ \\
        --output results.json \\
        --top_k 5

Output (single video, ``--top_k 5``)::

    {
      "video": "path/to/video.mp4",
      "predictions": [
        {"rank": 1, "label": "book",     "score": 0.912},
        {"rank": 2, "label": "read",     "score": 0.871},
        {"rank": 3, "label": "library",  "score": 0.843},
        {"rank": 4, "label": "study",    "score": 0.811},
        {"rank": 5, "label": "paper",    "score": 0.798}
      ],
      "inference_time_ms": 47.3
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from config.base_config import Config
from models.sign_model import SignModel
from utils.embedding_utils import load_embeddings_and_vocab

log = logging.getLogger(__name__)


# =============================================================================
# Device selection
# =============================================================================


def _resolve_device(device_str: str) -> torch.device:
    """Resolve an inference device string to a :class:`torch.device`.

    Args:
        device_str: One of ``"cuda"``, ``"cpu"``, or ``"auto"``.
            ``"auto"`` selects CUDA if available, otherwise CPU.

    Returns:
        Resolved :class:`torch.device`.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# =============================================================================
# Video preprocessing for inference (live-feed / .mp4)
# =============================================================================


def preprocess_video(
    video_path: str | Path,
    cfg: Config,
    device: torch.device,
) -> Tensor | None:
    """Decode a .mp4 file and preprocess for inference.

    Decodes frames via decord (preferred) or OpenCV (fallback), resamples
    uniformly to ``cfg.dataset.num_frames``, and applies the same deterministic
    val-mode pipeline used during training: resize → centre crop → normalise.
    No stochastic augmentation is applied.

    This function is the live-feed path.  For webcam input, capture frames
    externally into ``(T, H, W, 3)`` uint8 and call
    :func:`~data.dataset.wlasl_dataset.preprocess_live_frames` directly.

    Args:
        video_path: Path to the ``.mp4`` video file.  The file lives at
            ``WLASL300/<class_idx>/<video_id>.mp4`` for dataset videos.
        cfg: Full project config.
        device: Target device for the output tensor.

    Returns:
        Float32 tensor ``(1, 3, T, H, W)`` ready for the model,
        or ``None`` if the video could not be decoded.
    """
    from dataset.data.wlasl_dataset import preprocess_live_frames

    video_path = str(video_path)
    ds_cfg = cfg.dataset

    frames = _decode_frames(video_path, ds_cfg.num_frames, ds_cfg.loop_short_videos)
    if frames is None:
        return None

    return preprocess_live_frames(frames, cfg, device=device)


def _decode_frames(
    video_path: str,
    num_frames: int,
    loop_short: bool,
) -> np.ndarray | None:
    """Decode a video file into a uint8 frame array.

    Tries decord first (fast), falls back to OpenCV if decord is unavailable
    or raises an error.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample uniformly.
        loop_short: Loop clips shorter than ``num_frames``.

    Returns:
        ``(num_frames, H, W, 3)`` uint8 NumPy array, or ``None`` on failure.
    """
    try:
        return _decode_with_decord(video_path, num_frames, loop_short)
    except ImportError:
        pass
    except Exception as exc:
        log.warning("decord failed for %s: %s — trying cv2", video_path, exc)
    try:
        return _decode_with_cv2(video_path, num_frames, loop_short)
    except Exception as exc:
        log.error("Failed to decode %s: %s", video_path, exc)
        return None


def _decode_with_decord(
    video_path: str,
    num_frames: int,
    loop_short: bool,
) -> np.ndarray:
    """Decode video using decord (preferred for speed).

    Args:
        video_path: Path to the video file.
        num_frames: Desired output frame count.
        loop_short: Loop short clips.

    Returns:
        ``(num_frames, H, W, 3)`` uint8 array.

    Raises:
        ImportError: If decord is not installed.
        RuntimeError: If the video cannot be opened.
    """
    from decord import VideoReader, cpu  # type: ignore[import]

    vr = VideoReader(video_path, ctx=cpu(0))
    indices = _sample_indices(len(vr), num_frames, loop_short)
    return vr.get_batch(indices.tolist()).asnumpy()


def _decode_with_cv2(
    video_path: str,
    num_frames: int,
    loop_short: bool,
) -> np.ndarray:
    """Decode video using OpenCV (fallback).

    Args:
        video_path: Path to the video file.
        num_frames: Desired output frame count.
        loop_short: Loop short clips.

    Returns:
        ``(num_frames, H, W, 3)`` uint8 array (RGB).

    Raises:
        RuntimeError: If the video cannot be opened.
    """
    import cv2  # type: ignore[import]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_indices(total, num_frames, loop_short)
    all_frames: list[np.ndarray] = []
    prev: np.ndarray | None = None
    fi, pi = 0, 0

    while cap.isOpened() and pi < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        prev = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        while pi < len(indices) and indices[pi] == fi:
            all_frames.append(prev)
            pi += 1
        fi += 1
    cap.release()

    if prev is not None:
        while len(all_frames) < num_frames:
            all_frames.append(prev)

    return np.stack(all_frames[:num_frames], axis=0)


def _sample_indices(total: int, num_frames: int, loop_short: bool) -> np.ndarray:
    """Compute uniformly spaced frame indices.

    Args:
        total: Total frames available.
        num_frames: Desired sample count.
        loop_short: Loop when ``total < num_frames``.

    Returns:
        Integer NumPy array of length ``num_frames``.
    """
    if total >= num_frames:
        return np.linspace(0, total - 1, num=num_frames, dtype=int)
    if loop_short:
        base = np.arange(total)
        return np.tile(base, (num_frames // total) + 1)[:num_frames]
    return np.concatenate([np.arange(total), np.full(num_frames - total, total - 1)])


# =============================================================================
# Single-video prediction
# =============================================================================


def predict_single(
    video_path: str | Path,
    model: SignModel,
    class_embeddings: Tensor,
    vocab: list[str],
    cfg: Config,
    device: torch.device,
) -> dict:
    """Run top-k prediction on a single video file.

    Args:
        video_path: Path to the ``.mp4`` video file.
        model: Loaded :class:`~models.sign_model.SignModel` in eval mode.
        class_embeddings: Class embedding matrix ``(C, D)`` on ``device``.
        vocab: Ordered class name list of length ``C``.
        cfg: Full project config.
        device: Compute device.

    Returns:
        Dict with keys:

        - ``"video"``: Input video path string.
        - ``"predictions"``: List of ``top_k`` dicts, each with
          ``"rank"``, ``"label"``, and ``"score"`` (cosine similarity).
        - ``"inference_time_ms"``: Wall-clock inference time in milliseconds.
        - ``"error"``: Error message string if preprocessing failed,
          else absent.
    """
    video_path = str(video_path)
    top_k = cfg.inference.top_k

    t0 = time.perf_counter()

    # Preprocess
    video_tensor = preprocess_video(video_path, cfg, device)
    if video_tensor is None:
        return {
            "video": video_path,
            "predictions": [],
            "inference_time_ms": 0.0,
            "error": "Video decoding failed — check the file is a valid .mp4",
        }

    # Forward pass + nearest-neighbour retrieval
    with torch.no_grad():
        top_indices, top_scores = model.predict_topk(video_tensor, class_embeddings, k=top_k)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    predictions = [
        {
            "rank": rank + 1,
            "label": vocab[top_indices[0, rank].item()],
            "score": round(float(top_scores[0, rank].item()), 4),
        }
        for rank in range(top_k)
    ]

    return {
        "video": video_path,
        "predictions": predictions,
        "inference_time_ms": round(elapsed_ms, 2),
    }


# =============================================================================
# Batch directory prediction
# =============================================================================


def predict_directory(
    video_dir: str | Path,
    model: SignModel,
    class_embeddings: Tensor,
    vocab: list[str],
    cfg: Config,
    device: torch.device,
    output_path: str | Path | None = None,
) -> list[dict]:
    """Run top-k prediction on all ``.mp4`` files in a directory.

    Processes videos one at a time (not batched) so that decoding errors
    on individual files do not abort the entire run.  For GPU-accelerated
    batch inference, consider batching the preprocessed tensors using
    :func:`predict_batch`.

    Args:
        video_dir: Directory containing ``.mp4`` video files.
            Searched recursively.
        model: Loaded :class:`~models.sign_model.SignModel` in eval mode.
        class_embeddings: Class embedding matrix ``(C, D)`` on ``device``.
        vocab: Ordered class name list.
        cfg: Full project config.
        device: Compute device.
        output_path: Optional path to write results JSON.  If ``None``,
            results are returned but not written to disk.

    Returns:
        List of result dicts, one per video file (same format as
        :func:`predict_single`).
    """
    video_dir = Path(video_dir)
    mp4_files = sorted(video_dir.rglob("*.mp4"))

    if not mp4_files:
        log.warning("No .mp4 files found in %s", video_dir)
        return []

    log.info("Running inference on %d videos in %s …", len(mp4_files), video_dir)

    results: list[dict] = []
    errors = 0

    for i, video_path in enumerate(mp4_files, 1):
        result = predict_single(
            video_path=video_path,
            model=model,
            class_embeddings=class_embeddings,
            vocab=vocab,
            cfg=cfg,
            device=device,
        )
        results.append(result)

        if "error" in result:
            errors += 1
            log.warning("[%d/%d] ERROR  %s", i, len(mp4_files), result["error"])
        else:
            top1 = result["predictions"][0]["label"] if result["predictions"] else "?"
            score = result["predictions"][0]["score"] if result["predictions"] else 0.0
            log.info(
                "[%d/%d] %s → %s (%.3f)  [%.1f ms]",
                i,
                len(mp4_files),
                video_path.name,
                top1,
                score,
                result["inference_time_ms"],
            )

    log.info(
        "Inference complete: %d/%d successful, %d errors",
        len(results) - errors,
        len(results),
        errors,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info("Results written → %s", output_path)

    return results


# =============================================================================
# GPU-batched inference helper
# =============================================================================


def predict_batch(
    video_paths: list[str | Path],
    model: SignModel,
    class_embeddings: Tensor,
    vocab: list[str],
    cfg: Config,
    device: torch.device,
) -> list[dict]:
    """Run batched GPU inference over a list of video paths.

    Pre-processes all videos, stacks successfully decoded clips into a
    single batch tensor, runs one forward pass, and returns results.
    Clips that fail decoding are returned with an error entry without
    blocking the rest of the batch.

    This is faster than :func:`predict_directory` when all videos fit in
    GPU memory and the preprocessing overhead is already paid.

    Args:
        video_paths: List of paths to ``.mp4`` video files.
        model: Loaded :class:`~models.sign_model.SignModel` in eval mode.
        class_embeddings: Class embedding matrix ``(C, D)`` on ``device``.
        vocab: Ordered class name list.
        cfg: Full project config.
        device: Compute device.

    Returns:
        List of result dicts in the same format as :func:`predict_single`,
        in the same order as ``video_paths``.
    """
    top_k = cfg.inference.top_k
    results: list[dict | None] = [None] * len(video_paths)

    # Preprocess all videos; track which failed
    tensors: list[Tensor] = []
    valid_indices: list[int] = []

    for i, vp in enumerate(video_paths):
        tensor = preprocess_video(vp, cfg, device)
        if tensor is None:
            results[i] = {
                "video": str(vp),
                "predictions": [],
                "inference_time_ms": 0.0,
                "error": "Video decoding failed",
            }
        else:
            tensors.append(tensor)
            valid_indices.append(i)

    if not tensors:
        return [r for r in results if r is not None]

    # Stack into a single batch and run one forward pass
    batch = torch.cat(tensors, dim=0)  # (B, 3, T, H, W)
    t0 = time.perf_counter()

    with torch.no_grad():
        top_indices, top_scores = model.predict_topk(batch, class_embeddings, k=top_k)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_clip_ms = elapsed_ms / len(tensors)

    for batch_pos, original_idx in enumerate(valid_indices):
        predictions = [
            {
                "rank": rank + 1,
                "label": vocab[top_indices[batch_pos, rank].item()],
                "score": round(float(top_scores[batch_pos, rank].item()), 4),
            }
            for rank in range(top_k)
        ]
        results[original_idx] = {
            "video": str(video_paths[original_idx]),
            "predictions": predictions,
            "inference_time_ms": round(per_clip_ms, 2),
        }

    return [r for r in results if r is not None]


# =============================================================================
# Result formatting helpers
# =============================================================================


def format_result(result: dict, verbose: bool = True) -> str:
    """Format a single prediction result as a human-readable string.

    Args:
        result: Result dict from :func:`predict_single`.
        verbose: If ``True``, show all predictions.  If ``False``, show
            only the top-1 prediction.

    Returns:
        Multi-line formatted string.
    """
    lines = [f"Video: {result['video']}"]

    if "error" in result:
        lines.append(f"  ERROR: {result['error']}")
        return "\n".join(lines)

    lines.append(f"  Inference time: {result['inference_time_ms']:.1f} ms")

    preds = result.get("predictions", [])
    if not preds:
        lines.append("  No predictions (empty result)")
        return "\n".join(lines)

    if not verbose:
        p = preds[0]
        lines.append(f"  Prediction: {p['label']}  (score={p['score']:.4f})")
        return "\n".join(lines)

    lines.append("  Top predictions:")
    for p in preds:
        bar = "█" * int(p["score"] * 20)
        lines.append(f"    {p['rank']:>2}. {p['label']:<20}  {p['score']:.4f}  {bar}")

    return "\n".join(lines)


def compute_accuracy_from_results(
    results: list[dict],
    ground_truth: dict[str, str],
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Compute top-k accuracy from batch results against ground-truth labels.

    Useful for evaluating a directory of labelled test videos without
    running the full training evaluation loop.

    Args:
        results: List of result dicts from :func:`predict_directory`.
        ground_truth: Dict mapping video path string to ground-truth label.
        topk: Tuple of k-values to evaluate.

    Returns:
        Dict mapping ``"top{k}"`` to float accuracy in ``[0, 1]``.
    """
    correct: dict[int, int] = {k: 0 for k in topk}
    total = 0

    for result in results:
        video = result.get("video", "")
        true_label = ground_truth.get(video)
        if true_label is None:
            continue

        preds = result.get("predictions", [])
        total += 1
        for k in topk:
            top_k_labels = [p["label"] for p in preds[:k]]
            if true_label in top_k_labels:
                correct[k] += 1

    if total == 0:
        return {f"top{k}": 0.0 for k in topk}

    return {f"top{k}": correct[k] / total for k in topk}


# =============================================================================
# CLI argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the inference script.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Run sign language recognition inference on video clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input — mutually exclusive: single video or directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a single .mp4 video file to predict.",
    )
    input_group.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory containing .mp4 files for batch inference.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a trained model checkpoint (.pt file). "
            "Defaults to trained_models/best/checkpoint.pt if not provided."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the project YAML configuration file.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Number of top predictions to return.  Overrides config value.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=("Path to write results as JSON.  " "Defaults to stdout if not provided."),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "auto"],
        help="Compute device for inference.  Overrides config value.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help=(
            "Use GPU-batched inference for --video_dir "
            "(faster but requires all clips to fit in GPU memory)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed per-video prediction output to stdout.",
    )
    return parser.parse_args()


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    """Entry point for the inference script.

    Loads the config and checkpoint, resolves the device, and runs
    single-video or batch-directory inference based on the provided flags.

    Called by ``uv run python inference/inference.py`` or the ``infer``
    console script defined in ``pyproject.toml``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # ------------------------------------------------------------------ #
    # Load config
    # ------------------------------------------------------------------ #
    cfg = Config.from_yaml(args.config)

    # Apply CLI overrides
    if args.top_k is not None:
        import dataclasses

        new_inference = dataclasses.replace(cfg.inference, top_k=args.top_k)
        cfg = dataclasses.replace(cfg, inference=new_inference)

    device_str = args.device or cfg.inference.device
    device = _resolve_device(device_str)
    log.info("Inference device: %s", device)

    # ------------------------------------------------------------------ #
    # Load checkpoint
    # ------------------------------------------------------------------ #
    ckpt_path = args.checkpoint or cfg.inference.default_checkpoint
    if not Path(ckpt_path).exists():
        log.error(
            "Checkpoint not found: %s\n" "Train a model first with: uv run python train/train.py",
            ckpt_path,
        )
        raise SystemExit(1)

    log.info("Loading checkpoint from %s …", ckpt_path)
    model, epoch, metrics = SignModel.load_checkpoint(ckpt_path, cfg, device=str(device))
    model.eval()
    log.info(
        "Checkpoint loaded — epoch=%d  val_top1=%.4f",
        epoch,
        metrics.get("top1", 0.0),
    )

    # ------------------------------------------------------------------ #
    # Load class embeddings and vocab
    # ------------------------------------------------------------------ #
    class_embeddings, vocab = load_embeddings_and_vocab(
        cfg.paths.embeddings_file,
        cfg.paths.vocab_file,
        device=str(device),
    )
    log.info(
        "Class embeddings loaded — %d classes, dim=%d",
        class_embeddings.shape[0],
        class_embeddings.shape[1],
    )

    # ------------------------------------------------------------------ #
    # Run inference
    # ------------------------------------------------------------------ #
    if args.video:
        # Single-video mode
        result = predict_single(
            video_path=args.video,
            model=model,
            class_embeddings=class_embeddings,
            vocab=vocab,
            cfg=cfg,
            device=device,
        )
        print(format_result(result, verbose=args.verbose or True))

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                json.dump(result, f, indent=2)
            log.info("Result written → %s", args.output)

    else:
        # Batch-directory mode
        if args.batch:
            mp4_files = sorted(Path(args.video_dir).rglob("*.mp4"))
            log.info(
                "Batched inference on %d files (batch_size=%d) …",
                len(mp4_files),
                cfg.inference.batch_size,
            )
            all_results: list[dict] = []
            for i in range(0, len(mp4_files), cfg.inference.batch_size):
                chunk = mp4_files[i : i + cfg.inference.batch_size]
                batch_results = predict_batch(
                    video_paths=chunk,
                    model=model,
                    class_embeddings=class_embeddings,
                    vocab=vocab,
                    cfg=cfg,
                    device=device,
                )
                all_results.extend(batch_results)
                if args.verbose:
                    for r in batch_results:
                        print(format_result(r, verbose=False))
        else:
            all_results = predict_directory(
                video_dir=args.video_dir,
                model=model,
                class_embeddings=class_embeddings,
                vocab=vocab,
                cfg=cfg,
                device=device,
                output_path=args.output,
            )
            if args.verbose:
                for r in all_results:
                    print(format_result(r, verbose=False))

        # Write output JSON if batch mode (predict_directory writes its own)
        if args.batch and args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                json.dump(all_results, f, indent=2)
            log.info("Batch results written → %s", args.output)

        # Summary stats
        successful = [r for r in all_results if "error" not in r]
        if successful:
            mean_ms = np.mean([r["inference_time_ms"] for r in successful])
            log.info(
                "Summary: %d/%d successful  mean_latency=%.1f ms",
                len(successful),
                len(all_results),
                mean_ms,
            )


if __name__ == "__main__":
    main()
