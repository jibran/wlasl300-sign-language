"""Video loading, sampling, and quality audit utilities.

This module provides helper functions for decoding video files, validating
dataset integrity, and computing video-level statistics.  These utilities
are used by the annotation pipeline, the dataset class, and the training
loop's quality checks.

Example::

    from utils.video_utils import probe_video, audit_video_dir

    info = probe_video("data/raw/book/00620.mp4")
    print(info)
    # VideoInfo(path=..., num_frames=29, fps=25.0, width=256, height=256,
    #           duration_secs=1.16, is_valid=True, error=None)

    report = audit_video_dir("data/raw/", num_workers=4)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class VideoInfo:
    """Metadata and validity flag for a single video file.

    Attributes:
        path: Absolute path to the video file.
        num_frames: Total number of decoded frames.
        fps: Frames per second reported by the container.
        width: Frame width in pixels.
        height: Frame height in pixels.
        duration_secs: Clip duration in seconds.
        is_valid: ``True`` if the file could be opened and decoded.
        error: Error message if ``is_valid`` is ``False``, else ``None``.
    """

    path: str
    num_frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    duration_secs: float = 0.0
    is_valid: bool = True
    error: str | None = None


@dataclass
class AuditReport:
    """Summary report produced by :func:`audit_video_dir`.

    Attributes:
        total: Total number of ``.mp4`` files scanned.
        valid: Number of successfully decoded files.
        corrupt: Number of files that could not be decoded.
        too_short: Number of files below the minimum duration threshold.
        too_long: Number of files above the maximum duration threshold.
        missing_labels: Labels present in the directory but absent from
            the expected vocabulary.
        corrupt_paths: List of paths to corrupt / undecodable files.
        duration_stats: Dict with ``min``, ``mean``, ``max``, ``std``
            of clip durations in seconds (computed over valid clips).
        frame_count_stats: Dict with ``min``, ``mean``, ``max``, ``std``
            of frame counts (computed over valid clips).
        per_class_counts: Dict mapping label string to number of valid clips.
    """

    total: int = 0
    valid: int = 0
    corrupt: int = 0
    too_short: int = 0
    too_long: int = 0
    missing_labels: list[str] = field(default_factory=list)
    corrupt_paths: list[str] = field(default_factory=list)
    duration_stats: dict[str, float] = field(default_factory=dict)
    frame_count_stats: dict[str, float] = field(default_factory=dict)
    per_class_counts: dict[str, int] = field(default_factory=dict)


# =============================================================================
# Single-file probe
# =============================================================================


def probe_video(path: str | Path) -> VideoInfo:
    """Probe a single video file and return its metadata.

    Uses ``decord`` for fast probing without full frame decode.  Falls back
    to ``cv2.VideoCapture`` if ``decord`` is unavailable.

    Args:
        path: Path to the video file to probe.

    Returns:
        :class:`VideoInfo` populated with file metadata.  If the file cannot
        be opened, ``is_valid=False`` and ``error`` contains the message.
    """
    path = str(path)
    try:
        return _probe_with_decord(path)
    except ImportError:
        pass
    try:
        return _probe_with_cv2(path)
    except Exception as exc:
        return VideoInfo(path=path, is_valid=False, error=str(exc))


def _probe_with_decord(path: str) -> VideoInfo:
    """Probe a video file using decord (fast, preferred).

    Args:
        path: Path to the video file.

    Returns:
        :class:`VideoInfo` with metadata populated from the decord reader.

    Raises:
        ImportError: If decord is not installed.
        RuntimeError: If the file cannot be opened.
    """
    from decord import VideoReader, cpu  # type: ignore[import]

    try:
        vr = VideoReader(path, ctx=cpu(0))
    except Exception as exc:
        return VideoInfo(path=path, is_valid=False, error=str(exc))

    num_frames = len(vr)
    fps = float(vr.get_avg_fps())
    duration = num_frames / fps if fps > 0 else 0.0

    # Decode a single frame to get spatial dimensions
    try:
        frame = vr[0].asnumpy()
        height, width = frame.shape[:2]
    except Exception:
        height, width = 0, 0

    return VideoInfo(
        path=path,
        num_frames=num_frames,
        fps=fps,
        width=width,
        height=height,
        duration_secs=duration,
        is_valid=True,
    )


def _probe_with_cv2(path: str) -> VideoInfo:
    """Probe a video file using OpenCV (fallback).

    Args:
        path: Path to the video file.

    Returns:
        :class:`VideoInfo` with metadata populated from ``cv2.VideoCapture``.

    Raises:
        ImportError: If OpenCV is not installed.
        RuntimeError: If the file cannot be opened.
    """
    import cv2  # type: ignore[import]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return VideoInfo(path=path, is_valid=False, error="cv2 could not open file")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = num_frames / fps if fps > 0 else 0.0
    cap.release()

    return VideoInfo(
        path=path,
        num_frames=num_frames,
        fps=fps,
        width=width,
        height=height,
        duration_secs=duration,
        is_valid=True,
    )


# =============================================================================
# Directory-level audit
# =============================================================================


def audit_video_dir(
    raw_dir: str | Path,
    expected_vocab: list[str] | None = None,
    min_duration: float = 0.5,
    max_duration: float = 10.0,
    num_workers: int = 4,
) -> AuditReport:
    """Scan a raw video directory and produce a quality audit report.

    Walks ``raw_dir`` recursively, probes every ``.mp4`` file, and aggregates
    statistics on duration, frame count, and per-class counts.

    Args:
        raw_dir: Root directory of the raw Kaggle WLASL download.  Expected
            layout: ``raw_dir/<label>/<video_id>.mp4``.
        expected_vocab: Optional list of expected class names.  Labels found
            in ``raw_dir`` that are absent from ``expected_vocab`` are
            reported in :attr:`AuditReport.missing_labels`.
        min_duration: Clips shorter than this (seconds) are flagged as
            ``too_short``.
        max_duration: Clips longer than this (seconds) are flagged as
            ``too_long``.
        num_workers: Number of threads for parallel probing.  Use ``1``
            for sequential (easier debugging).

    Returns:
        :class:`AuditReport` with full dataset statistics.

    Example::

        report = audit_video_dir("data/raw/", expected_vocab=vocab, num_workers=4)
        print(f"Valid: {report.valid}/{report.total}")
        print(f"Corrupt: {report.corrupt_paths}")
    """
    raw_dir = Path(raw_dir)
    mp4_files = sorted(raw_dir.rglob("*.mp4"))

    if not mp4_files:
        log.warning("No .mp4 files found in %s", raw_dir)
        return AuditReport()

    log.info("Auditing %d video files in %s …", len(mp4_files), raw_dir)

    # Probe files in parallel
    infos: list[VideoInfo] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(probe_video, f): f for f in mp4_files}
        for i, future in enumerate(as_completed(futures), 1):
            infos.append(future.result())
            if i % 500 == 0:
                log.info("  Probed %d / %d files …", i, len(mp4_files))

    # Aggregate statistics
    report = AuditReport(total=len(infos))
    durations: list[float] = []
    frame_counts: list[int] = []

    for info in infos:
        # Infer label from parent directory name
        label = Path(info.path).parent.name.lower()
        if not info.is_valid:
            report.corrupt += 1
            report.corrupt_paths.append(info.path)
            continue

        report.valid += 1
        durations.append(info.duration_secs)
        frame_counts.append(info.num_frames)
        report.per_class_counts[label] = report.per_class_counts.get(label, 0) + 1

        if info.duration_secs < min_duration:
            report.too_short += 1
            log.debug("Too short (%.2fs): %s", info.duration_secs, info.path)
        elif info.duration_secs > max_duration:
            report.too_long += 1
            log.debug("Too long (%.2fs): %s", info.duration_secs, info.path)

    # Duration statistics
    if durations:
        dur_arr = np.array(durations)
        report.duration_stats = {
            "min": float(dur_arr.min()),
            "mean": float(dur_arr.mean()),
            "max": float(dur_arr.max()),
            "std": float(dur_arr.std()),
        }
        fc_arr = np.array(frame_counts)
        report.frame_count_stats = {
            "min": int(fc_arr.min()),
            "mean": float(fc_arr.mean()),
            "max": int(fc_arr.max()),
            "std": float(fc_arr.std()),
        }

    # Check for labels missing from expected vocabulary
    if expected_vocab is not None:
        found_labels = set(report.per_class_counts.keys())
        expected_set = {w.lower() for w in expected_vocab}
        report.missing_labels = sorted(expected_set - found_labels)

    log.info(
        "Audit complete — valid=%d  corrupt=%d  too_short=%d  too_long=%d",
        report.valid,
        report.corrupt,
        report.too_short,
        report.too_long,
    )
    if report.missing_labels:
        log.warning("Labels in vocab but not on disk: %s", report.missing_labels)

    return report


# =============================================================================
# Frame sampling helpers
# =============================================================================


def uniform_sample_indices(
    total_frames: int,
    num_frames: int,
    start: int = 0,
    end: int | None = None,
) -> np.ndarray:
    """Compute uniformly-spaced frame indices within a segment.

    Args:
        total_frames: Total number of frames in the video.
        num_frames: Number of indices to return.
        start: First valid frame index (0-indexed, inclusive).
        end: Last valid frame index (0-indexed, inclusive).  Defaults to
            ``total_frames - 1``.

    Returns:
        Integer NumPy array of shape ``(num_frames,)`` with indices in
        ``[start, end]``, evenly spaced.

    Raises:
        ValueError: If ``num_frames < 1`` or ``start > end``.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be ≥ 1, got {num_frames}")
    if end is None:
        end = total_frames - 1
    if start > end:
        raise ValueError(f"start ({start}) must be ≤ end ({end})")

    return np.linspace(start, end, num=num_frames, dtype=int)


def loop_pad_indices(
    total_frames: int,
    num_frames: int,
    start: int = 0,
    end: int | None = None,
) -> np.ndarray:
    """Compute frame indices with cyclic looping for short clips.

    When the segment ``[start, end]`` has fewer frames than ``num_frames``,
    the available indices are tiled cyclically to fill the output.

    Args:
        total_frames: Total number of frames in the video.
        num_frames: Desired output length.
        start: First valid frame index (0-indexed, inclusive).
        end: Last valid frame index (0-indexed, inclusive).

    Returns:
        Integer NumPy array of shape ``(num_frames,)`` with indices in
        ``[start, end]``, looped if necessary.
    """
    if end is None:
        end = total_frames - 1
    start = max(0, start)
    end = min(end, total_frames - 1)

    base = np.arange(start, end + 1)
    segment_len = len(base)

    if segment_len >= num_frames:
        return np.linspace(start, end, num=num_frames, dtype=int)

    repeats = (num_frames // segment_len) + 1
    return np.tile(base, repeats)[:num_frames]
