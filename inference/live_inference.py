"""Live webcam inference for WLASL300 sign language recognition.

Captures frames from a webcam in a rolling buffer, runs the trained
:class:`~models.sign_model.SignModel` on each completed clip, and overlays
the top-k predictions on the live OpenCV window.

The capture loop accumulates ``cfg.dataset.num_frames`` frames (16 by
default) into a sliding window buffer.  Every ``--stride`` new frames the
buffer is flushed through the model and the prediction is refreshed on
screen.  A lower stride gives more frequent predictions at the cost of more
GPU/CPU time; a higher stride reduces prediction frequency.

**Usage**::

    uv run python inference/live_inference.py \\
        --checkpoint trained_models/best/checkpoint.pt

**Controls** (while the window is focused):

- ``q`` — quit
- ``r`` — reset the frame buffer (use between signs)
- ``s`` — save the current buffer as ``live_capture_<timestamp>.npy``

**Requirements**:

- A webcam accessible via OpenCV (``cv2.VideoCapture(0)``).
- A trained checkpoint — run ``train/train.py`` first.
- ``opencv-python-headless`` is sufficient for capture; ``opencv-python``
  is needed if you want the GUI window (default).  Install the full
  variant if the window does not open::

      pip install opencv-python

"""

from __future__ import annotations

import argparse
import collections
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from config.base_config import Config
from dataset.data.wlasl_dataset import preprocess_live_frames
from models.sign_model import SignModel
from utils.embedding_utils import load_embeddings_and_vocab

log = logging.getLogger(__name__)

# Overlay appearance
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (80, 200, 80)
_WHITE = (240, 240, 240)
_BLACK = (20, 20, 20)
_YELLOW = (40, 210, 210)
_BAR_BG = (50, 50, 50)


# =============================================================================
# Frame buffer
# =============================================================================


class FrameBuffer:
    """Fixed-length rolling buffer of raw webcam frames.

    Stores the most recent ``capacity`` frames as uint8 numpy arrays.
    When full, the oldest frame is discarded on each new push (sliding
    window semantics).

    Args:
        capacity: Number of frames to retain (== ``cfg.dataset.num_frames``).
    """

    def __init__(self, capacity: int) -> None:
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, frame: np.ndarray) -> None:
        """Append one ``(H, W, 3)`` uint8 RGB frame.

        Args:
            frame: Single video frame in RGB order.
        """
        self._buf.append(frame)

    def full(self) -> bool:
        """Return ``True`` when the buffer holds exactly ``capacity`` frames.

        Returns:
            Boolean fullness flag.
        """
        return len(self._buf) == self.capacity

    def as_array(self) -> np.ndarray:
        """Return buffer contents as ``(T, H, W, 3)`` uint8 array.

        Returns:
            Stacked frame array ready for :func:`preprocess_live_frames`.
        """
        return np.stack(list(self._buf), axis=0)

    def reset(self) -> None:
        """Clear all frames from the buffer."""
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


# =============================================================================
# Overlay rendering
# =============================================================================


def _draw_overlay(
    frame: np.ndarray,
    predictions: list[dict],
    fps: float,
    buffer_pct: float,
    is_predicting: bool,
) -> np.ndarray:
    """Draw prediction overlay and HUD onto a BGR frame in-place.

    Renders a semi-transparent panel in the top-left corner showing
    top-k predictions with confidence bars, a buffer fill indicator, and
    the current inference FPS.

    Args:
        frame: BGR uint8 frame from OpenCV (modified in-place).
        predictions: List of ``{"rank", "label", "score"}`` dicts from
            :meth:`~models.sign_model.SignModel.predict_topk`.
        fps: Current inference frames-per-second estimate.
        buffer_pct: Buffer fill fraction in [0, 1].
        is_predicting: Whether inference is currently running.

    Returns:
        The annotated frame (same object as input).
    """
    h, w = frame.shape[:2]
    panel_w = 320
    panel_h = 40 + len(predictions) * 36 + 50
    panel = frame[:panel_h, :panel_w].copy()
    cv2.rectangle(panel, (0, 0), (panel_w, panel_h), _BLACK, -1)
    cv2.addWeighted(panel, 0.65, frame[:panel_h, :panel_w], 0.35, 0, frame[:panel_h, :panel_w])

    y = 24
    cv2.putText(frame, "WLASL300 live", (10, y), _FONT, 0.55, _WHITE, 1, cv2.LINE_AA)
    fps_label = f"{fps:.1f} pred/s" if fps > 0 else "warming up..."
    cv2.putText(frame, fps_label, (panel_w - 110, y), _FONT, 0.45, _YELLOW, 1, cv2.LINE_AA)

    y += 10
    for pred in predictions:
        y += 26
        label = pred["label"]
        score = float(pred["score"])
        rank = pred["rank"]

        colour = _GREEN if rank == 1 else _WHITE
        cv2.putText(frame, f"{rank}. {label}", (10, y), _FONT, 0.52, colour, 1, cv2.LINE_AA)

        bar_x, bar_y = 180, y - 12
        bar_max = 120
        bar_fill = int(bar_max * max(0.0, min(1.0, score)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max, bar_y + 10), _BAR_BG, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + 10), _GREEN, -1)
        cv2.putText(
            frame,
            f"{score:.2f}",
            (bar_x + bar_max + 5, y - 2),
            _FONT,
            0.4,
            _WHITE,
            1,
            cv2.LINE_AA,
        )

    y += 20
    buf_label = "buffer: "
    cv2.putText(frame, buf_label, (10, y), _FONT, 0.42, _WHITE, 1, cv2.LINE_AA)
    bx = 80
    cv2.rectangle(frame, (bx, y - 10), (bx + 160, y), _BAR_BG, -1)
    fill_w = int(160 * buffer_pct)
    fill_col = _GREEN if buffer_pct >= 1.0 else _YELLOW
    cv2.rectangle(frame, (bx, y - 10), (bx + fill_w, y), fill_col, -1)

    if is_predicting:
        cv2.putText(frame, "running...", (bx + 168, y - 2), _FONT, 0.38, _YELLOW, 1, cv2.LINE_AA)

    controls = "[q] quit  [r] reset  [s] save"
    cv2.putText(frame, controls, (10, h - 12), _FONT, 0.38, _WHITE, 1, cv2.LINE_AA)

    return frame


# =============================================================================
# Inference worker
# =============================================================================


def run_inference(
    frames: np.ndarray,
    model: SignModel,
    class_embeddings: torch.Tensor,
    vocab: list[str],
    cfg: Config,
    device: torch.device,
    top_k: int,
) -> list[dict]:
    """Run a single forward pass and return top-k predictions.

    Args:
        frames: ``(T, H, W, 3)`` uint8 numpy array from the frame buffer.
        model: Loaded :class:`~models.sign_model.SignModel` in eval mode.
        class_embeddings: ``(num_classes, embedding_dim)`` float32 tensor.
        vocab: Ordered list of class label strings.
        cfg: Project config.
        device: Inference device.
        top_k: Number of predictions to return.

    Returns:
        List of ``{"rank", "label", "score"}`` dicts ordered by score.
    """
    video_tensor = preprocess_live_frames(frames, cfg, device=device)

    with torch.no_grad():
        predictions = model.predict_topk(
            video_tensor,
            class_embeddings=class_embeddings,
            vocab=vocab,
            k=top_k,
        )

    return predictions[0] if predictions else []


# =============================================================================
# Main capture loop
# =============================================================================


def live_inference(
    checkpoint: str,
    config: str,
    camera_id: int,
    top_k: int,
    stride: int,
    device_str: str,
    width: int,
    height: int,
) -> None:
    """Run the live webcam inference loop.

    Opens the webcam, loads the model, and enters the capture-predict-display
    loop until the user presses ``q``.

    Args:
        checkpoint: Path to the trained ``.pt`` checkpoint file.
        config: Path to ``config/config.yaml``.
        camera_id: OpenCV camera index (0 = default webcam).
        top_k: Number of top predictions to display.
        stride: Number of new frames between successive predictions.
        device_str: ``"cuda"``, ``"cpu"``, or ``"auto"``.
        width: Requested webcam capture width in pixels.
        height: Requested webcam capture height in pixels.

    Raises:
        SystemExit: If the checkpoint or webcam cannot be opened.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ------------------------------------------------------------------ #
    # Resolve device
    # ------------------------------------------------------------------ #
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    log.info("Inference device: %s", device)

    # ------------------------------------------------------------------ #
    # Load config
    # ------------------------------------------------------------------ #
    cfg = Config.from_yaml(config)
    num_frames = cfg.dataset.num_frames

    # ------------------------------------------------------------------ #
    # Load checkpoint
    # ------------------------------------------------------------------ #
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        log.error(
            "Checkpoint not found: %s\nTrain a model first: uv run python train/train.py",
            ckpt_path,
        )
        raise SystemExit(1)

    log.info("Loading checkpoint: %s", ckpt_path)
    model, epoch, metrics = SignModel.load_checkpoint(str(ckpt_path), cfg, device=str(device))
    model.eval()
    log.info("Checkpoint loaded — epoch=%d  val_top1=%.4f", epoch, metrics.get("top1", 0.0))

    # ------------------------------------------------------------------ #
    # Load class embeddings + vocab
    # ------------------------------------------------------------------ #
    class_embeddings, vocab = load_embeddings_and_vocab(
        cfg.paths.embeddings_file,
        cfg.paths.vocab_file,
        device=str(device),
    )
    log.info("Classes: %d  ·  embedding dim: %d", len(vocab), class_embeddings.shape[1])

    # ------------------------------------------------------------------ #
    # Open webcam
    # ------------------------------------------------------------------ #
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        log.error("Cannot open camera %d", camera_id)
        raise SystemExit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Camera opened: %dx%d  camera_id=%d", actual_w, actual_h, camera_id)
    log.info("Buffer size: %d frames  ·  prediction stride: %d frames", num_frames, stride)
    log.info("Controls: [q] quit  [r] reset buffer  [s] save buffer")

    # ------------------------------------------------------------------ #
    # Capture / predict loop
    # ------------------------------------------------------------------ #
    buffer = FrameBuffer(capacity=num_frames)
    predictions: list[dict] = []
    frames_since_pred = 0
    pred_times: collections.deque[float] = collections.deque(maxlen=10)
    is_predicting = False

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            log.warning("Failed to read frame from camera — retrying …")
            time.sleep(0.05)
            continue

        # Convert BGR → RGB and push to buffer
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        buffer.push(rgb_frame)
        frames_since_pred += 1

        # Run inference when buffer is full and stride has elapsed
        if buffer.full() and frames_since_pred >= stride:
            frames_since_pred = 0
            is_predicting = True
            t0 = time.perf_counter()

            try:
                predictions = run_inference(
                    frames=buffer.as_array(),
                    model=model,
                    class_embeddings=class_embeddings,
                    vocab=vocab,
                    cfg=cfg,
                    device=device,
                    top_k=top_k,
                )
            except Exception as exc:
                log.error("Inference error: %s", exc)
                predictions = []

            elapsed = time.perf_counter() - t0
            pred_times.append(elapsed)
            is_predicting = False

            if predictions:
                top = predictions[0]
                log.info(
                    "Prediction: %-15s  score=%.3f  inference=%.1fms",
                    top["label"],
                    top["score"],
                    elapsed * 1000,
                )

        # Compute rolling inference FPS
        fps = 1.0 / (sum(pred_times) / len(pred_times)) if pred_times else 0.0
        buffer_pct = len(buffer) / num_frames

        # Draw overlay
        display = _draw_overlay(
            bgr_frame.copy(),
            predictions,
            fps=fps,
            buffer_pct=buffer_pct,
            is_predicting=is_predicting,
        )

        cv2.imshow("WLASL300 — live sign language recognition", display)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            log.info("Quitting.")
            break
        elif key == ord("r"):
            buffer.reset()
            predictions = []
            log.info("Buffer reset.")
        elif key == ord("s"):
            ts = int(time.time())
            save_path = Path(f"live_capture_{ts}.npy")
            np.save(str(save_path), buffer.as_array())
            log.info("Buffer saved → %s  shape=%s", save_path, buffer.as_array().shape)

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    p = argparse.ArgumentParser(
        description="Live webcam sign language recognition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="trained_models/best/checkpoint.pt",
        help="Path to trained model checkpoint (.pt).",
    )
    p.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml.",
    )
    p.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (0 = default webcam).",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top predictions to display.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=16,
        help=(
            "Run inference every N new frames. "
            "Lower = more frequent predictions, higher CPU/GPU load. "
            "Higher = less frequent but cheaper."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Inference device.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested webcam capture width.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested webcam capture height.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for live inference.

    Parses CLI arguments and starts the webcam capture loop.
    """
    args = parse_args()
    live_inference(
        checkpoint=args.checkpoint,
        config=args.config,
        camera_id=args.camera,
        top_k=args.top_k,
        stride=args.stride,
        device_str=args.device,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
