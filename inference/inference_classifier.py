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
import math
import time
from pathlib import Path

import numpy as np
import torch

from config.base_config import Config
from dataset.data.wlasl_dataset import preprocess_live_frames
from models.sign_model_classifier import SignModelClassifier
from models.sign_model_deep import SignModelDeep
from models.sign_model_temporal import SignModelTemporal
from models.sign_model_linear import SignModelLinear


# =============================================================================
# Video decoding helpers  (inlined from inference.inference to avoid
# package-relative import issues when running the script directly)
# =============================================================================


def _sample_indices(total: int, num_frames: int, loop_short: bool) -> "np.ndarray":
    """Compute uniformly spaced frame indices.

    Args:
        total: Total frames available in the video.
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


def _decode_with_decord(
    video_path: str,
    num_frames: int,
    loop_short: bool,
) -> "np.ndarray":
    """Decode video using decord (preferred — faster than cv2).

    Args:
        video_path: Path to the video file.
        num_frames: Desired output frame count.
        loop_short: Loop short clips to reach ``num_frames``.

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
) -> "np.ndarray":
    """Decode video using OpenCV (fallback when decord is unavailable).

    Args:
        video_path: Path to the video file.
        num_frames: Desired output frame count.
        loop_short: Loop short clips to reach ``num_frames``.

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


def _decode_frames(
    video_path: str,
    num_frames: int,
    loop_short: bool,
) -> "np.ndarray | None":
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
# Results overlay video renderer
# =============================================================================


def render_overlay_video(
    video_path: str,
    predictions: list[dict],
    out_path: str,
    cfg: object,
) -> None:
    """Re-render a video with top-k prediction results burned into every frame.

    Draws a full-width dark panel at the top of the frame containing:
    - One row per prediction spanning the full frame width
    - Label on the left, a filled confidence bar in the middle, score on the right
    - Green highlight for rank-1, blue-grey for the rest

    Uses only OpenCV — no additional dependencies.

    Args:
        video_path: Path to the original input ``.mp4`` file.
        predictions: List of ``{"rank", "label", "score"}`` dicts ordered by
            score descending, as returned by ``predict_single``.
        out_path: Destination path for the annotated output ``.mp4``.
        cfg: Project config (unused; kept for signature compatibility).

    Raises:
        RuntimeError: If the video cannot be opened or the writer fails.
    """
    import cv2  # type: ignore[import]

    # ── Open source video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for overlay: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # ── Writer ───────────────────────────────────────────────────────────────
    out_path = str(out_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (src_w, src_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter for: {out_path}")

    # ── Layout — full frame width, top-left origin ───────────────────────────
    n_preds = len(predictions)
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = max(6, src_w // 80)  # ~8 px on 640-wide video

    # Row height scales with frame height; each row holds one prediction
    row_h = max(24, src_h // (n_preds * 3 + 4))
    thickness = max(1, math.ceil(row_h / 28))

    # Font scales derived from row height
    label_scale = row_h / 48.0
    score_scale = row_h / 52.0

    # Panel spans full frame width, sits at the very top
    panel_w = src_w
    panel_h = n_preds * row_h + pad
    panel_x = 0
    panel_y = 0

    # Score column width: enough for "0.9999" at score_scale
    (score_tw, _), _ = cv2.getTextSize("0.9999", font, score_scale, thickness)
    score_col_w = score_tw + pad * 2

    # Bar column: between label and score
    bar_col_x = panel_w // 2  # bar starts at horizontal midpoint
    bar_col_w = panel_w - bar_col_x - score_col_w - pad * 2
    bar_h = max(6, row_h // 3)

    # Label column: left half minus padding
    label_col_w = bar_col_x - pad * 2

    # Colour palette (BGR)
    BG_DARK = (15, 15, 15)
    BG_TOP = (28, 42, 28)  # faint green tint for rank-1
    BORDER = (70, 70, 70)
    LABEL_TOP = (255, 255, 255)
    LABEL_REST = (190, 190, 190)
    BAR_BG = (45, 45, 45)
    BAR_GREEN = (60, 180, 70)  # rank-1
    BAR_BLUE = (160, 130, 90)  # others (warm blue-grey in BGR)
    SCORE_TOP = (140, 230, 140)  # bright green
    SCORE_REST = (150, 150, 150)

    def _draw_panel(frame: np.ndarray) -> np.ndarray:
        """Burn the full-width top panel onto a single frame (in-place)."""

        # Draw each row with its own background before blending
        overlay = frame.copy()

        for i in range(n_preds):
            row_y0 = panel_y + i * row_h
            row_y1 = row_y0 + row_h
            bg = BG_TOP if i == 0 else BG_DARK
            cv2.rectangle(overlay, (panel_x, row_y0), (panel_x + panel_w, row_y1), bg, -1)

        # Blend the background rows (72 % opaque)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Bottom border of the whole panel
        cv2.line(
            frame,
            (0, panel_y + panel_h),
            (src_w, panel_y + panel_h),
            BORDER,
            1,
        )

        # Render each prediction row
        for i, pred in enumerate(predictions):
            row_y0 = panel_y + i * row_h
            is_top = i == 0
            score = float(pred["score"])

            # Vertical centre for text baseline
            text_y = row_y0 + row_h - pad

            # ── Label (left column) ──────────────────────────────────────────
            label_str = f"#{pred['rank']}  {pred['label']}"
            # Truncate if it overflows the label column
            (tw, _), _ = cv2.getTextSize(label_str, font, label_scale, thickness)
            while tw > label_col_w and len(label_str) > 4:
                label_str = label_str[:-1]
                (tw, _), _ = cv2.getTextSize(label_str + "…", font, label_scale, thickness)
            if tw > label_col_w:
                label_str = label_str + "…"

            cv2.putText(
                frame,
                label_str,
                (panel_x + pad, text_y),
                font,
                label_scale,
                LABEL_TOP if is_top else LABEL_REST,
                thickness,
                cv2.LINE_AA,
            )

            # Thin row separator
            if i > 0:
                cv2.line(
                    frame,
                    (0, row_y0),
                    (src_w, row_y0),
                    BORDER,
                    1,
                )

            # ── Confidence bar (centre column) ────────────────────────────────
            bar_x = bar_col_x
            bar_y = row_y0 + (row_h - bar_h) // 2
            filled = max(2, int(bar_col_w * score))

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_col_w, bar_y + bar_h),
                BAR_BG,
                -1,
            )
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + filled, bar_y + bar_h),
                BAR_GREEN if is_top else BAR_BLUE,
                -1,
            )

            # ── Score (right column) ─────────────────────────────────────────
            score_str = f"{score:.4f}"
            (sw, _), _ = cv2.getTextSize(score_str, font, score_scale, thickness)
            score_x = src_w - sw - pad
            cv2.putText(
                frame,
                score_str,
                (score_x, text_y),
                font,
                score_scale,
                SCORE_TOP if is_top else SCORE_REST,
                thickness,
                cv2.LINE_AA,
            )

        return frame

    # ── Write every frame ────────────────────────────────────────────────────
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _draw_panel(frame)
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()

    if written == 0:
        raise RuntimeError(f"No frames written — source had {total_frames} frames")

    log.info(
        "Overlay video → %s  (%d frames  %.0f fps  %dx%d)",
        out_path,
        written,
        fps,
        src_w,
        src_h,
    )


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
    p.add_argument(
        "--overlay",
        type=str,
        default=None,
        metavar="OUT.mp4",
        help="Render a copy of the input video with predictions burned in. "
        "Only valid with --video (single-video mode). "
        "If omitted the overlay is written alongside the source as "
        "<source_stem>_overlay.mp4 when --video is used.",
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

        # Overlay: render annotated video if requested (or auto-named)
        if "predictions" in result:
            overlay_path = args.overlay
            if overlay_path is None:
                src = Path(args.video)
                overlay_path = str(src.parent / f"{src.stem}_overlay{src.suffix}")
            try:
                render_overlay_video(
                    video_path=args.video,
                    predictions=result["predictions"],
                    out_path=overlay_path,
                    cfg=cfg,
                )
            except Exception as exc:
                log.warning("Overlay rendering failed: %s", exc)

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
