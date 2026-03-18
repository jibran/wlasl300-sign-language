"""Shared pytest fixtures for WLASL300 test suite.

All fixtures in this file are automatically available to every test module
without explicit imports.  Fixtures are scoped at the ``"session"`` level
where construction is expensive (e.g. loading a Config) and at the
``"function"`` level where state mutation is possible.

Example::

    def test_projection_head_shape(cfg, device):
        from models.projection_head import ProjectionHead
        head = ProjectionHead(
            input_dim=cfg.model.backbone_output_dim,
            hidden_dim=cfg.model.projection_hidden_dim,
            output_dim=cfg.model.embedding_dim,
        )
        x = torch.randn(4, cfg.model.backbone_output_dim)
        out = head(x)
        assert out.shape == (4, cfg.model.embedding_dim)
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

from config.base_config import Config
from models.projection_head import ProjectionHead

# =============================================================================
# Config fixture
# =============================================================================


@pytest.fixture(scope="session")
def cfg() -> Config:
    """Return a fully populated Config loaded from config/config.yaml.

    Scoped to the test session so the YAML is only parsed once.

    Returns:
        :class:`~config.base_config.Config` instance.
    """
    from config.base_config import Config

    return Config.from_yaml("config/config.yaml")


# =============================================================================
# Device fixture
# =============================================================================


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return the best available torch device for tests.

    Prefers CUDA if available, otherwise CPU.  GPU-dependent tests should
    use this fixture and call ``pytest.skip`` if CUDA is unavailable.

    Returns:
        :class:`torch.device` — ``"cuda"`` or ``"cpu"``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Minimal config fixture (no disk I/O)
# =============================================================================


@pytest.fixture(scope="session")
def minimal_cfg() -> Config:
    """Return a Config built from defaults — no YAML file required.

    Useful for unit tests that need a config but should not depend on the
    presence of ``config/config.yaml`` on disk.

    Returns:
        :class:`~config.base_config.Config` with all default values.
    """
    from config.base_config import Config

    return Config()


# =============================================================================
# Dummy video tensor fixture
# =============================================================================


@pytest.fixture
def dummy_video(cfg: Config) -> torch.Tensor:
    """Return a random ``(1, 3, T, H, W)`` float32 video tensor.

    Simulates one preprocessed video clip on CPU, ready for the model
    forward pass.

    Args:
        cfg: Project config providing ``num_frames`` and ``frame_size``.

    Returns:
        Float32 tensor of shape
        ``(1, 3, cfg.dataset.num_frames, cfg.dataset.frame_size, cfg.dataset.frame_size)``.
    """
    T = cfg.dataset.num_frames
    H = W = cfg.dataset.frame_size
    return torch.randn(1, 3, T, H, W)


@pytest.fixture
def dummy_video_batch(cfg: Config) -> torch.Tensor:
    """Return a random ``(4, 3, T, H, W)`` float32 video batch.

    Args:
        cfg: Project config providing spatial and temporal dimensions.

    Returns:
        Float32 tensor of shape ``(4, 3, T, H, W)``.
    """
    B, T, H, W = 4, cfg.dataset.num_frames, cfg.dataset.frame_size, cfg.dataset.frame_size
    return torch.randn(B, 3, T, H, W)


# =============================================================================
# Dummy embeddings and vocab fixtures
# =============================================================================


@pytest.fixture(scope="session")
def dummy_vocab() -> list[str]:
    """Return a small vocabulary list of 10 class names for fast testing.

    Returns:
        List of 10 ASL word strings.
    """
    return [
        "book",
        "drink",
        "computer",
        "before",
        "chair",
        "go",
        "clothes",
        "who",
        "candy",
        "cousin",
    ]


@pytest.fixture(scope="session")
def dummy_embeddings(dummy_vocab: list[str]) -> torch.Tensor:
    """Return a random L2-normalised embedding matrix for the dummy vocab.

    Shape: ``(len(dummy_vocab), 300)``, float32, unit vectors.

    Args:
        dummy_vocab: List of class names from the ``dummy_vocab`` fixture.

    Returns:
        Float32 tensor of shape ``(10, 300)``.
    """
    import torch.nn.functional as F

    emb = torch.randn(len(dummy_vocab), 300)
    return F.normalize(emb, p=2, dim=-1)


# =============================================================================
# Temporary annotation directory fixture
# =============================================================================


@pytest.fixture
def tmp_annotation_dir(
    dummy_vocab: list[str], dummy_embeddings: torch.Tensor, tmp_path: pathlib.Path
) -> pathlib.Path:
    """Create a temporary annotations directory with all required files.

    Writes minimal ``annotations.json``, ``vocab.json``, ``splits.json``,
    and ``word2vec_embeddings.npy`` so that
    :class:`~data.dataset.wlasl_dataset.WLASL300Dataset` can be instantiated
    without the real dataset on disk.

    Args:
        dummy_vocab: Fixture providing a 10-word vocabulary.
        dummy_embeddings: Fixture providing a ``(10, 300)`` embedding matrix.
        tmp_path: pytest built-in temporary directory.

    Returns:
        :class:`~pathlib.Path` to the temporary annotations directory.
    """
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()

    # vocab.json
    vocab_path = ann_dir / "vocab.json"
    vocab_path.write_text(json.dumps(dummy_vocab))

    # word2vec_embeddings.npy
    emb_path = ann_dir / "word2vec_embeddings.npy"
    np.save(str(emb_path), dummy_embeddings.numpy())

    base = "nonexistent"
    # annotations.json — 5 dummy records per class split across train/val/test
    records = []
    splits_cycle = ["train", "train", "train", "val", "test"]
    for label_idx, label in enumerate(dummy_vocab):
        for i, split in enumerate(splits_cycle):
            video_id = f"{label_idx:03d}{i:02d}"
            records.append(
                {
                    "video_id": video_id,
                    "frames_dir": f"/{base}/preprocessing/{split}/frames/{label_idx}/{video_id}",
                    "video_path": f"/{base}/WLASL300/{label_idx}/{video_id}.mp4",
                    "label": label,
                    "label_idx": label_idx,
                    "class_idx": label_idx,
                    "split": split,
                    "num_frames": 16,
                    "frame_pattern": f"{label_idx}_{{}}.jpg",
                }
            )

    ann_path = ann_dir / "annotations.json"
    ann_path.write_text(json.dumps(records, indent=2))

    # splits.json
    splits = {"train": [], "val": [], "test": []}
    for r in records:
        splits[r["split"]].append(r["video_id"])
    splits_path = ann_dir / "splits.json"
    splits_path.write_text(json.dumps(splits, indent=2))

    return ann_dir


# =============================================================================
# Model fixtures
# =============================================================================


@pytest.fixture
def projection_head(cfg: Config) -> ProjectionHead:
    """Return a freshly initialised ProjectionHead on CPU.

    Args:
        cfg: Project config.

    Returns:
        :class:`~models.projection_head.ProjectionHead` instance.
    """
    from models.projection_head import ProjectionHead

    return ProjectionHead(
        input_dim=cfg.model.backbone_output_dim,
        hidden_dim=cfg.model.projection_hidden_dim,
        output_dim=cfg.model.embedding_dim,
        dropout=cfg.model.dropout,
        l2_normalize=cfg.model.l2_normalize,
    )
