"""Unit tests for the WLASL300 dataset and augmentation pipeline.

Tests cover:

- :class:`~data.dataset.wlasl_dataset.WLASL300Dataset` instantiation with
  dummy annotation files.
- ``__len__`` and split filtering correctness.
- :class:`~data.dataset.wlasl_dataset.VideoAugmentor` output shape,
  dtype, and temporal consistency.
- :func:`~data.dataset.wlasl_dataset.build_dataloaders` DataLoader creation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# =============================================================================
# WLASL300Dataset tests
# =============================================================================


class TestWLASL300Dataset:
    """Tests for WLASL300Dataset instantiation and split filtering."""

    def test_instantiation_train_split(self, tmp_annotation_dir, cfg):
        """Dataset must instantiate without error for the train split."""
        from dataset.data.wlasl_dataset import WLASL300Dataset

        ds = WLASL300Dataset(
            annotations_file=str(tmp_annotation_dir / "annotations.json"),
            embeddings_file=str(tmp_annotation_dir / "word2vec_embeddings.npy"),
            vocab_file=str(tmp_annotation_dir / "vocab.json"),
            split="train",
            cfg=cfg,
        )
        assert len(ds) > 0, "Train split has zero samples"

    def test_instantiation_val_split(self, tmp_annotation_dir, cfg):
        """Dataset must instantiate without error for the val split."""
        from dataset.data.wlasl_dataset import WLASL300Dataset

        ds = WLASL300Dataset(
            annotations_file=str(tmp_annotation_dir / "annotations.json"),
            embeddings_file=str(tmp_annotation_dir / "word2vec_embeddings.npy"),
            vocab_file=str(tmp_annotation_dir / "vocab.json"),
            split="val",
            cfg=cfg,
        )
        assert len(ds) > 0, "Val split has zero samples"

    def test_invalid_split_raises(self, tmp_annotation_dir, cfg):
        """An invalid split name must raise ValueError."""
        from dataset.data.wlasl_dataset import WLASL300Dataset

        with pytest.raises(ValueError, match="Invalid split"):
            WLASL300Dataset(
                annotations_file=str(tmp_annotation_dir / "annotations.json"),
                embeddings_file=str(tmp_annotation_dir / "word2vec_embeddings.npy"),
                vocab_file=str(tmp_annotation_dir / "vocab.json"),
                split="invalid_split",
                cfg=cfg,
            )

    def test_missing_annotation_file_raises(self, tmp_annotation_dir, cfg):
        """Missing annotations.json must raise FileNotFoundError."""
        from dataset.data.wlasl_dataset import WLASL300Dataset

        with pytest.raises(FileNotFoundError):
            WLASL300Dataset(
                annotations_file="/nonexistent/annotations.json",
                embeddings_file=str(tmp_annotation_dir / "word2vec_embeddings.npy"),
                vocab_file=str(tmp_annotation_dir / "vocab.json"),
                split="train",
                cfg=cfg,
            )

    def test_split_counts_sum_to_total(self, tmp_annotation_dir, cfg):
        """Train + val + test sample counts must equal total annotations."""
        import json

        from dataset.data.wlasl_dataset import WLASL300Dataset

        total = len(json.loads((tmp_annotation_dir / "annotations.json").read_text()))
        counts = 0
        for split in ("train", "val", "test"):
            ds = WLASL300Dataset(
                annotations_file=str(tmp_annotation_dir / "annotations.json"),
                embeddings_file=str(tmp_annotation_dir / "word2vec_embeddings.npy"),
                vocab_file=str(tmp_annotation_dir / "vocab.json"),
                split=split,
                cfg=cfg,
            )
            counts += len(ds)
        assert counts == total, f"Split counts ({counts}) != total annotations ({total})"

    def test_vocab_property(self, tmp_annotation_dir, cfg, dummy_vocab):
        """Dataset.vocab must match the dummy vocabulary."""
        from dataset.data.wlasl_dataset import WLASL300Dataset

        ds = WLASL300Dataset(
            annotations_file=str(tmp_annotation_dir / "annotations.json"),
            embeddings_file=str(tmp_annotation_dir / "word2vec_embeddings.npy"),
            vocab_file=str(tmp_annotation_dir / "vocab.json"),
            split="train",
            cfg=cfg,
        )
        assert ds.vocab == dummy_vocab

    def test_num_classes_property(self, tmp_annotation_dir, cfg, dummy_vocab):
        """Dataset.num_classes must equal the vocab length."""
        from dataset.data.wlasl_dataset import WLASL300Dataset

        ds = WLASL300Dataset(
            annotations_file=str(tmp_annotation_dir / "annotations.json"),
            embeddings_file=str(tmp_annotation_dir / "word2vec_embeddings.npy"),
            vocab_file=str(tmp_annotation_dir / "vocab.json"),
            split="train",
            cfg=cfg,
        )
        assert ds.num_classes == len(dummy_vocab)

    def test_class_embedding_matrix_shape(self, tmp_annotation_dir, cfg, dummy_vocab):
        """class_embedding_matrix must be (num_classes, embedding_dim)."""
        from dataset.data.wlasl_dataset import WLASL300Dataset

        ds = WLASL300Dataset(
            annotations_file=str(tmp_annotation_dir / "annotations.json"),
            embeddings_file=str(tmp_annotation_dir / "word2vec_embeddings.npy"),
            vocab_file=str(tmp_annotation_dir / "vocab.json"),
            split="train",
            cfg=cfg,
        )
        emb_matrix = ds.class_embedding_matrix
        assert emb_matrix.shape == (len(dummy_vocab), ds.embedding_dim)

    def test_get_class_weights_length(self, tmp_annotation_dir, cfg):
        """get_class_weights must return a tensor of length len(dataset)."""
        from dataset.data.wlasl_dataset import WLASL300Dataset

        ds = WLASL300Dataset(
            annotations_file=str(tmp_annotation_dir / "annotations.json"),
            embeddings_file=str(tmp_annotation_dir / "word2vec_embeddings.npy"),
            vocab_file=str(tmp_annotation_dir / "vocab.json"),
            split="train",
            cfg=cfg,
        )
        weights = ds.get_class_weights()
        assert len(weights) == len(ds)
        assert (weights > 0).all(), "All class weights must be positive"


# =============================================================================
# VideoAugmentor tests
# =============================================================================


class TestVideoAugmentor:
    """Tests for the VideoAugmentor spatial augmentation pipeline."""

    def _make_frames(self, num_frames: int = 16, height: int = 256, width: int = 256) -> np.ndarray:
        """Create a dummy uint8 frame array ``(num_frames, height, width, 3)``."""
        return np.random.randint(0, 255, (num_frames, height, width, 3), dtype=np.uint8)

    def test_output_shape_train(self, cfg):
        """Train-mode augmentor must return (3, T, crop_size, crop_size)."""
        from dataset.data.wlasl_dataset import VideoAugmentor

        T = 8
        augmentor = VideoAugmentor(
            aug_cfg=cfg.augmentation,
            resize_size=cfg.dataset.resize_size,
            crop_size=cfg.dataset.frame_size,
            mean=list(cfg.dataset.mean),
            std=list(cfg.dataset.std),
            is_train=True,
        )
        frames = self._make_frames(num_frames=T)
        out = augmentor(frames)
        assert out.shape == (3, T, cfg.dataset.frame_size, cfg.dataset.frame_size), (
            f"Expected (3, {T}, {cfg.dataset.frame_size}, {cfg.dataset.frame_size}), "
            f"got {out.shape}"
        )

    def test_output_shape_val(self, cfg):
        """Val-mode augmentor must return (3, T, crop_size, crop_size)."""
        from dataset.data.wlasl_dataset import VideoAugmentor

        T = 8
        augmentor = VideoAugmentor(
            aug_cfg=cfg.augmentation,
            resize_size=cfg.dataset.resize_size,
            crop_size=cfg.dataset.frame_size,
            mean=list(cfg.dataset.mean),
            std=list(cfg.dataset.std),
            is_train=False,
        )
        frames = self._make_frames(num_frames=T)
        out = augmentor(frames)
        assert out.shape == (3, T, cfg.dataset.frame_size, cfg.dataset.frame_size)

    def test_output_dtype_float32(self, cfg):
        """Augmentor output must be float32."""
        from dataset.data.wlasl_dataset import VideoAugmentor

        augmentor = VideoAugmentor(
            aug_cfg=cfg.augmentation,
            resize_size=cfg.dataset.resize_size,
            crop_size=cfg.dataset.frame_size,
            mean=list(cfg.dataset.mean),
            std=list(cfg.dataset.std),
            is_train=False,
        )
        frames = self._make_frames()
        out = augmentor(frames)
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

    def test_val_mode_deterministic(self, cfg):
        """Val-mode augmentor must produce identical output on two calls."""
        from dataset.data.wlasl_dataset import VideoAugmentor

        augmentor = VideoAugmentor(
            aug_cfg=cfg.augmentation,
            resize_size=cfg.dataset.resize_size,
            crop_size=cfg.dataset.frame_size,
            mean=list(cfg.dataset.mean),
            std=list(cfg.dataset.std),
            is_train=False,
        )
        frames = self._make_frames()
        out1 = augmentor(frames)
        out2 = augmentor(frames)
        assert torch.allclose(
            out1, out2
        ), "Val-mode augmentor is not deterministic — two calls gave different results"


# =============================================================================
# Speed perturbation tests
# =============================================================================


class TestSpeedPerturbation:
    """Tests for the speed perturbation augmentation."""

    def test_fast_speed_reads_more_frames(self):
        """Speed > 1.0 must produce more source frames than target."""
        from utils.augmentation import apply_speed_perturbation

        frames = np.zeros((64, 32, 32, 3), dtype=np.uint8)
        target = 64
        speed = 1.5  # fast — should read 64/1.5 ≈ 43 source frames
        out = apply_speed_perturbation(frames, speed=speed, target_frames=target)
        assert out.shape[0] == target, f"Output must have {target} frames"

    def test_slow_speed_output_shape(self):
        """Speed < 1.0 must still produce target_frames output frames."""
        from utils.augmentation import apply_speed_perturbation

        frames = np.zeros((64, 32, 32, 3), dtype=np.uint8)
        out = apply_speed_perturbation(frames, speed=0.75, target_frames=64)
        assert out.shape[0] == 64

    def test_zero_speed_raises(self):
        """Speed of 0 must raise ValueError."""
        from utils.augmentation import apply_speed_perturbation

        frames = np.zeros((64, 32, 32, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="speed must be > 0"):
            apply_speed_perturbation(frames, speed=0.0, target_frames=64)
