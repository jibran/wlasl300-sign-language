"""Annotation pipeline for the WLASL300 sign language dataset.

This script reads ``folder2label_str.txt``, discovers all pre-extracted JPG
frame clips under ``preprocessing/<split>/frames/``, verifies every expected
frame file exists, and pre-computes L2-normalised Word2Vec embeddings.

Dataset layout on disk::

    WLASL300/                          <- raw .mp4 files (inference only)
        <class_idx>/
            <video_id>.mp4

    preprocessing/                     <- pre-extracted frames (training)
        <split>/frames/<class_idx>/<video_id>/<class_idx>_0.jpg .. _15.jpg

    folder2label_str.txt               <- "0 about\n1 accident\n..."

Each clip contains exactly 16 frames at 256x256 pixels.
The frame filename pattern is ``<class_idx>_{0..15}.jpg``.

Outputs written to ``--out_dir`` (default ``dataset/annotations/``):

- ``annotations.json``  — one record per verified clip
- ``vocab.json``        — ordered label list (index == class_idx)
- ``splits.json``       — {train/val/test: [video_id, ...]}
- ``word2vec_embeddings.npy`` — (num_classes, 300) float32 L2-normalised
- ``oov_report.txt``    — OOV words and missing clip counts

Usage::

    python dataset/annotations/build_annotations.py \\
        --preprocessing_dir  dataset/raw/preprocessing \\
        --folder2label       dataset/raw/folder2label_str.txt \\
        --word2vec_bin       trained_models/embeddings/GoogleNews-vectors-negative300.bin \\
        --out_dir            dataset/annotations
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_SPLITS = ("train", "val", "test")


# =============================================================================
# Step 1 — Load label mapping
# =============================================================================


def load_folder2label(folder2label_path: Path) -> dict[int, str]:
    """Load ``folder2label_str.txt`` into a class-index-to-label dict.

    Each line has the format ``<class_idx> <label_str>``, e.g.::

        0 about
        1 accident

    Args:
        folder2label_path: Path to ``folder2label_str.txt``.

    Returns:
        Dict mapping integer class index to lowercase label string.

    Raises:
        FileNotFoundError: If the mapping file does not exist.
        ValueError: If any line cannot be parsed as ``<int> <str>``.
    """
    if not folder2label_path.exists():
        raise FileNotFoundError(
            f"folder2label_str.txt not found: {folder2label_path}\n"
            "This file should be in the root of your dataset directory."
        )

    mapping: dict[int, str] = {}
    with folder2label_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Unparseable line {lineno} in {folder2label_path}: {line!r}\n"
                    "Expected format: '<class_idx> <label_str>'"
                )
            mapping[int(parts[0])] = parts[1].strip().lower()

    log.info("Loaded %d class labels from %s", len(mapping), folder2label_path)
    return mapping


# =============================================================================
# Step 2 — Discover and verify clips
# =============================================================================


def discover_clips(
    preprocessing_dir: Path,
    wlasl_dir: Path,
    label_map: dict[int, str],
    num_frames: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Walk preprocessing/ and verify all frame files exist for each clip.

    For each split directory the expected structure is::

        <split>/frames/<class_idx>/<video_id>/<class_idx>_<n>.jpg  (n=0..15)

    Also records the corresponding raw ``.mp4`` path for inference.

    Args:
        preprocessing_dir: Root of the pre-extracted frame tree.
        wlasl_dir: Root of the raw ``WLASL300/`` video directory.
        label_map: Dict from :func:`load_folder2label`.
        num_frames: Expected frames per clip (16).

    Returns:
        A tuple of:
            - ``clips``: Verified clip record dicts.
            - ``incomplete``: Video IDs with missing frame files.
    """
    clips: list[dict[str, Any]] = []
    incomplete: list[str] = []
    seen_ids: set[str] = set()

    for split in _SPLITS:
        frames_root = preprocessing_dir / split / "frames"
        if not frames_root.exists():
            log.warning("Split directory not found, skipping: %s", frames_root)
            continue

        class_dirs = sorted(d for d in frames_root.iterdir() if d.is_dir() and d.name.isdigit())

        for class_dir in class_dirs:
            class_idx = int(class_dir.name)
            label = label_map.get(class_idx)
            if label is None:
                log.warning("class_idx %d not in folder2label — skipping", class_idx)
                continue

            frame_pattern = f"{class_idx}_{{}}.jpg"

            for clip_dir in sorted(d for d in class_dir.iterdir() if d.is_dir()):
                video_id = clip_dir.name

                # Verify all 16 frames are present
                missing = [
                    i
                    for i in range(num_frames)
                    if not (clip_dir / frame_pattern.format(i)).exists()
                ]
                if missing:
                    log.debug(
                        "Incomplete clip %s/%s — missing frames: %s",
                        class_idx,
                        video_id,
                        missing,
                    )
                    incomplete.append(video_id)
                    continue

                # Deduplicate across splits
                if video_id in seen_ids:
                    log.debug("Duplicate video_id %s in %s — skipping", video_id, split)
                    continue
                seen_ids.add(video_id)

                # Resolve raw .mp4 path (may not exist for all clips)
                mp4_path = wlasl_dir / str(class_idx) / f"{video_id}.mp4"
                if not mp4_path.exists():
                    alt = wlasl_dir / str(class_idx) / f"{video_id.zfill(5)}.mp4"
                    if alt.exists():
                        mp4_path = alt

                clips.append(
                    {
                        "video_id": video_id,
                        "frames_dir": str(clip_dir),
                        "video_path": str(mp4_path),
                        "label": label,
                        "label_idx": class_idx,
                        "class_idx": class_idx,
                        "split": split,
                        "num_frames": num_frames,
                        "frame_pattern": frame_pattern,
                    }
                )

    log.info(
        "Clip discovery: %d complete, %d incomplete",
        len(clips),
        len(incomplete),
    )
    for s in _SPLITS:
        n = sum(1 for c in clips if c["split"] == s)
        log.info("  %-6s : %d clips", s, n)

    return clips, incomplete


# =============================================================================
# Step 3 — Build vocabulary
# =============================================================================


def build_vocab(label_map: dict[int, str]) -> list[str]:
    """Build a vocab list ordered by class index.

    ``vocab[i]`` is the label string for class index ``i``.

    Args:
        label_map: Dict mapping class index to label string.

    Returns:
        List of label strings of length ``max(label_map) + 1``.

    Raises:
        ValueError: If class indices are not contiguous from 0.
    """
    max_idx = max(label_map.keys())
    vocab: list[str] = []
    for idx in range(max_idx + 1):
        if idx not in label_map:
            raise ValueError(
                f"Class index {idx} missing from folder2label_str.txt. "
                f"Expected contiguous 0..{max_idx}."
            )
        vocab.append(label_map[idx])
    log.info("Vocabulary: %d classes", len(vocab))
    return vocab


# =============================================================================
# Step 4 — Assign label indices
# =============================================================================


def assign_label_indices(
    clips: list[dict[str, Any]],
    vocab: list[str],
) -> list[dict[str, Any]]:
    """Set ``label_idx`` on each clip to its position in ``vocab``.

    Since vocab is ordered by class_idx, this is an identity mapping but
    makes the alignment explicit and verifiable.

    Args:
        clips: Clip record dicts from :func:`discover_clips`.
        vocab: Ordered vocab list from :func:`build_vocab`.

    Returns:
        Same list with ``label_idx`` updated.
    """
    label_to_idx = {label: idx for idx, label in enumerate(vocab)}
    for clip in clips:
        clip["label_idx"] = label_to_idx[clip["label"]]
    return clips


# =============================================================================
# Step 5 — Build splits map
# =============================================================================


def build_splits_map(clips: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build a split map from clip records.

    Splits are taken directly from the filesystem layout — no re-splitting
    is performed since the dataset provides official splits.

    Args:
        clips: Clip records with ``split`` and ``video_id`` set.

    Returns:
        Dict mapping split name to list of ``video_id`` strings.
    """
    splits_map: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for clip in clips:
        splits_map[clip["split"]].append(clip["video_id"])
    log.info(
        "Splits — train: %d  val: %d  test: %d",
        len(splits_map["train"]),
        len(splits_map["val"]),
        len(splits_map["test"]),
    )
    return splits_map


# =============================================================================
# Step 6 — Word2Vec embeddings
# =============================================================================


def build_word2vec_embeddings(
    vocab: list[str],
    word2vec_bin: Path,
) -> tuple[np.ndarray, list[str]]:
    """Compute L2-normalised Word2Vec embeddings for all vocab words.

    Multi-token labels (e.g. "thank you") use the mean of available token
    vectors. OOV words receive a zero vector and are reported separately.

    Args:
        vocab: Ordered list of class label strings.
        word2vec_bin: Path to ``GoogleNews-vectors-negative300.bin``.

    Returns:
        A tuple of:
            - ``embeddings``: Float32 array ``(len(vocab), 300)``.
            - ``oov_words``: Words not found in the Word2Vec vocabulary.

    Raises:
        FileNotFoundError: If ``word2vec_bin`` does not exist.
        ImportError: If gensim is not installed.
    """
    try:
        from gensim.models import KeyedVectors
    except ImportError as exc:
        raise ImportError(
            "gensim is required for Word2Vec embedding generation.\n"
            "Install it manually:  pip install gensim==4.4.0"
        ) from exc

    if not word2vec_bin.exists():
        raise FileNotFoundError(
            f"Word2Vec binary not found: {word2vec_bin}\n"
            "Expected: trained_models/embeddings/GoogleNews-vectors-negative300.bin\n"
            "Download from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM"
        )

    log.info("Loading Word2Vec from %s (~30 s) …", word2vec_bin)
    wv = KeyedVectors.load_word2vec_format(str(word2vec_bin), binary=True)
    log.info("Word2Vec loaded — vocab size: %d", len(wv))

    D = wv.vector_size
    embeddings = np.zeros((len(vocab), D), dtype=np.float32)
    oov_words: list[str] = []

    for idx, word in enumerate(vocab):
        tokens = word.lower().split()
        vecs = []
        for t in tokens:
            if t in wv:
                vecs.append(wv[t])
            else:
                for sub in t.replace("'", "").replace("-", " ").split():
                    if sub in wv:
                        vecs.append(wv[sub])
        if vecs:
            vec = np.mean(vecs, axis=0).astype(np.float32)
            norm = np.linalg.norm(vec)
            embeddings[idx] = vec / norm if norm > 1e-8 else vec
        else:
            oov_words.append(word)
            log.warning("OOV: '%s' — embedding set to zero", word)

    log.info(
        "Embeddings: %d/%d resolved, %d OOV",
        len(vocab) - len(oov_words),
        len(vocab),
        len(oov_words),
    )
    return embeddings, oov_words


# =============================================================================
# Step 7 — Write outputs
# =============================================================================


def write_outputs(
    out_dir: Path,
    clips: list[dict[str, Any]],
    vocab: list[str],
    splits_map: dict[str, list[str]],
    embeddings: np.ndarray,
    oov_words: list[str],
    incomplete_clips: list[str],
) -> None:
    """Write all annotation output files to ``out_dir``.

    Creates five files: ``annotations.json``, ``vocab.json``,
    ``splits.json``, ``word2vec_embeddings.npy``, and ``oov_report.txt``.

    Args:
        out_dir: Destination directory.
        clips: Verified clip record dicts.
        vocab: Ordered vocab list.
        splits_map: Split-to-video-ID mapping.
        embeddings: Word2Vec embedding matrix.
        oov_words: OOV word list.
        incomplete_clips: Video IDs skipped due to missing frames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "annotations.json").write_text(
        json.dumps(clips, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Wrote annotations.json  (%d records)", len(clips))

    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Wrote vocab.json  (%d classes)", len(vocab))

    (out_dir / "splits.json").write_text(
        json.dumps(splits_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(
        "Wrote splits.json  (train=%d val=%d test=%d)",
        len(splits_map["train"]),
        len(splits_map["val"]),
        len(splits_map["test"]),
    )

    np.save(str(out_dir / "word2vec_embeddings.npy"), embeddings)
    log.info("Wrote word2vec_embeddings.npy  %s", embeddings.shape)

    with (out_dir / "oov_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Total vocab        : {len(vocab)}\n")
        f.write(f"Resolved           : {len(vocab) - len(oov_words)}\n")
        f.write(f"OOV words          : {len(oov_words)}\n")
        f.write(f"Incomplete clips   : {len(incomplete_clips)}\n\n")
        if oov_words:
            f.write("OOV list:\n")
            for w in sorted(oov_words):
                f.write(f"  - {w}\n")
        else:
            f.write("All words resolved successfully.\n")
    log.info("Wrote oov_report.txt")


# =============================================================================
# Step 8 — Summary
# =============================================================================


def print_summary(
    clips: list[dict[str, Any]],
    vocab: list[str],
    splits_map: dict[str, list[str]],
    embeddings: np.ndarray,
    oov_words: list[str],
) -> None:
    """Print a concise pipeline summary to the log.

    Args:
        clips: Verified clip records.
        vocab: Ordered vocab list.
        splits_map: Split map.
        embeddings: Embedding matrix.
        oov_words: OOV words.
    """
    counts = np.array(list(Counter(c["label_idx"] for c in clips).values()))
    log.info("=" * 50)
    log.info("  Pipeline summary")
    log.info("=" * 50)
    log.info("  Classes            : %d", len(vocab))
    log.info("  Total clips        : %d", len(clips))
    log.info(
        "  Train/Val/Test     : %d / %d / %d",
        len(splits_map["train"]),
        len(splits_map["val"]),
        len(splits_map["test"]),
    )
    if len(counts):
        log.info("  Clips/class  min   : %d", int(counts.min()))
        log.info("  Clips/class  mean  : %.1f", float(counts.mean()))
        log.info("  Clips/class  max   : %d", int(counts.max()))
    log.info("  Embedding shape    : %s", embeddings.shape)
    log.info("  OOV words          : %d", len(oov_words))
    log.info("=" * 50)


# =============================================================================
# CLI entry point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    p = argparse.ArgumentParser(
        description="Build WLASL300 annotations from pre-extracted JPG frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--preprocessing_dir",
        type=Path,
        default=Path("preprocessing"),
        help="Root of pre-extracted JPG frames  (<split>/frames/<class_idx>/<video_id>/).",
    )
    p.add_argument(
        "--wlasl_dir",
        type=Path,
        default=Path("WLASL300"),
        help="Root of raw WLASL300 .mp4 videos  (<class_idx>/<video_id>.mp4).",
    )
    p.add_argument(
        "--folder2label",
        type=Path,
        default=Path("folder2label_str.txt"),
        help="Path to folder2label_str.txt  (format: '<class_idx> <label>').",
    )
    p.add_argument(
        "--word2vec_bin",
        type=Path,
        default=Path("trained_models/embeddings/GoogleNews-vectors-negative300.bin"),
        help="Path to GoogleNews-vectors-negative300.bin.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("dataset/annotations"),
        help="Directory to write annotation outputs.",
    )
    p.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Expected frames per clip.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (for reproducibility).",
    )
    return p.parse_args()


def main() -> None:
    """Run the full annotation pipeline end-to-end.

    Raises:
        FileNotFoundError: If label mapping or Word2Vec binary not found.
        SystemExit: If no complete clips are discovered.
    """
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    log.info("WLASL300 annotation pipeline starting")
    log.info("  preprocessing_dir : %s", args.preprocessing_dir)
    log.info("  wlasl_dir         : %s", args.wlasl_dir)
    log.info("  folder2label      : %s", args.folder2label)
    log.info("  word2vec_bin      : %s", args.word2vec_bin)
    log.info("  out_dir           : %s", args.out_dir)
    log.info("  num_frames        : %d", args.num_frames)

    label_map = load_folder2label(args.folder2label)
    clips, incomplete = discover_clips(
        args.preprocessing_dir, args.wlasl_dir, label_map, args.num_frames
    )
    if not clips:
        log.error("No complete clips found — check preprocessing_dir layout.")
        raise SystemExit(1)

    vocab = build_vocab(label_map)
    clips = assign_label_indices(clips, vocab)
    splits_map = build_splits_map(clips)
    embeddings, oov_words = build_word2vec_embeddings(vocab, args.word2vec_bin)

    write_outputs(
        out_dir=args.out_dir,
        clips=clips,
        vocab=vocab,
        splits_map=splits_map,
        embeddings=embeddings,
        oov_words=oov_words,
        incomplete_clips=incomplete,
    )
    print_summary(clips, vocab, splits_map, embeddings, oov_words)
    log.info("Pipeline complete. Outputs in: %s", args.out_dir)


if __name__ == "__main__":
    main()
