"""PyTorch Dataset and DataLoader factory for the WLASL300 dataset.

Loads pre-extracted JPG frames from the preprocessing directory for training
and evaluation.  For live-feed inference the model receives raw video frames
decoded from a webcam or .mp4 file — the same ``VideoAugmentor`` is used in
both paths so preprocessing is identical.

Frame layout on disk::

    preprocessing/<split>/frames/<class_idx>/<video_id>/<class_idx>_0.jpg
                                                         <class_idx>_1.jpg
                                                         ...
                                                         <class_idx>_15.jpg

Each clip has exactly **16 frames** at **256 x 256** pixels.  The augmentor
skips the resize step (``resize_size == frame_size``) and applies only spatial
crop / flip / colour jitter on the already-correct-size images.

For inference from live video or .mp4 files, call
:func:`preprocess_live_frames` which decodes a raw frame array and applies
the same deterministic val-mode pipeline.

Example::

    from config import Config
    from dataset.data import WLASL300Dataset, build_dataloaders

    cfg = Config.from_yaml("config/config.yaml")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    for video, embedding, label_idx in train_loader:
        # video:     (B, 3, 16, 256, 256)  float32
        # embedding: (B, 300)               float32 L2-normalised
        # label_idx: (B,)                   int64
        pass
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

_VALID_SPLITS = {"train", "val", "test"}


# =============================================================================
# JPG frame loading
# =============================================================================


def _load_jpg_frames(
    frames_dir: str | Path,
    frame_pattern: str,
    num_frames: int,
) -> np.ndarray:
    """Load a clip's pre-extracted JPG frames into a uint8 numpy array.

    Reads exactly ``num_frames`` files matching ``frame_pattern`` from
    ``frames_dir``.  The pattern uses Python format syntax with a single
    positional placeholder, e.g. ``"0_{}.jpg"``.

    Args:
        frames_dir: Directory containing the clip's JPG files.
        frame_pattern: Filename template, e.g. ``"0_{}.jpg"`` or
            ``"299_{}.jpg"``.  The ``{}`` is replaced with the frame index
            0, 1, …, num_frames-1.
        num_frames: Number of frames to load (16 for WLASL300).

    Returns:
        NumPy array of shape ``(num_frames, H, W, 3)``, dtype ``uint8``.

    Raises:
        RuntimeError: If any expected frame file cannot be opened.
    """
    frames_dir = Path(frames_dir)
    frames: list[np.ndarray] = []
    for i in range(num_frames):
        fname = frames_dir / frame_pattern.format(i)
        try:
            img = Image.open(fname).convert("RGB")
            frames.append(np.array(img, dtype=np.uint8))
        except Exception as exc:
            raise RuntimeError(f"Failed to load frame {fname}: {exc}") from exc
    return np.stack(frames, axis=0)  # (T, H, W, 3)


# =============================================================================
# VideoAugmentor — identical pipeline for JPG frames and live video
# =============================================================================


class VideoAugmentor:
    """Applies temporally-consistent spatial augmentations to a clip.

    Stochastic decisions (crop position, flip, colour jitter parameters)
    are sampled **once per clip** and applied **identically to every frame**
    so there are no inter-frame inconsistencies.

    When ``is_train=False`` (val / test / inference), only deterministic
    centre crop and ImageNet normalisation are applied.

    Since pre-extracted frames are already 256 x 256, the resize step is a
    no-op when ``resize_size == frame_size``.  For live-feed inference where
    frames may have arbitrary resolution the resize step is active.

    Args:
        aug_cfg: :class:`~config.base_config.AugmentationConfig`.
        resize_size: Shorter-side resize target (256 for pre-extracted frames).
        crop_size: Output spatial resolution (256).
        mean: Per-channel ImageNet normalisation mean.
        std: Per-channel ImageNet normalisation std.
        is_train: Enable stochastic augmentation.

    Example::

        augmentor = VideoAugmentor(cfg.augmentation, 256, 256,
                                   cfg.dataset.mean, cfg.dataset.std,
                                   is_train=True)
        frames_np = np.zeros((16, 256, 256, 3), dtype=np.uint8)
        video_tensor = augmentor(frames_np)   # (3, 16, 256, 256)
    """

    def __init__(
        self,
        aug_cfg: object,
        resize_size: int,
        crop_size: int,
        mean: list[float],
        std: list[float],
        is_train: bool,
    ) -> None:
        """Initialise the augmentor.

        Args:
            aug_cfg: Augmentation config dataclass.
            resize_size: Shorter-side resize target.
            crop_size: Output crop size.
            mean: Per-channel normalisation mean.
            std: Per-channel normalisation std.
            is_train: Enable stochastic transforms.
        """
        self._aug_cfg = aug_cfg
        self._resize_size = resize_size
        self._crop_size = crop_size
        self._is_train = is_train
        self._normalise = T.Normalize(mean=mean, std=std)
        # Only perform resize if the input may differ from target size
        self._needs_resize = resize_size != crop_size
        self._resize = T.Resize(resize_size, antialias=True)

    def __call__(self, frames: np.ndarray) -> Tensor:
        """Apply the augmentation pipeline to a raw uint8 frame array.

        Samples all stochastic parameters once, then applies them to every
        frame identically.  Pipeline:

        1. Resize shorter side (skipped if resize_size == crop_size)
        2. Random crop (train) / centre crop (val/test)
        3. Random horizontal flip (train only)
        4. Convert to float32 [0, 1]
        5. Colour jitter with fixed parameters per clip (train only)
        6. Gaussian noise (train only, if enabled)
        7. ImageNet normalisation

        Args:
            frames: ``(T, H, W, 3)`` uint8 NumPy array.

        Returns:
            Float32 :class:`torch.Tensor` of shape ``(3, T, H, W)``.
        """
        T_len = frames.shape[0]

        # --- Sample stochastic parameters once for the whole clip ---

        # Crop: sample from the actual frame dimensions
        first = torch.from_numpy(frames[0]).permute(2, 0, 1)  # (3, H, W)
        if self._needs_resize:
            first = self._resize(first)

        if self._is_train and self._aug_cfg.random_crop:
            crop_i, crop_j, crop_h, crop_w = T.RandomCrop.get_params(
                first, output_size=(self._crop_size, self._crop_size)
            )

            def _crop(img: Tensor) -> Tensor:
                return TF.crop(img, crop_i, crop_j, crop_h, crop_w)

        else:

            def _crop(img: Tensor) -> Tensor:
                return TF.center_crop(img, self._crop_size)

        do_flip = (
            self._is_train
            and self._aug_cfg.random_horizontal_flip
            and random.random() < self._aug_cfg.horizontal_flip_prob
        )

        # Sample color jitter parameters once per clip so every frame in the
        # clip gets the same colour transform (temporal consistency).
        # get_params returns (brightness, contrast, saturation, hue) floats —
        # these are applied per-frame via TF.adjust_* in the loop below.
        cj_params: tuple[float, float, float, float] | None = None
        if self._is_train and self._aug_cfg.color_jitter:
            cj = T.ColorJitter(
                brightness=self._aug_cfg.color_jitter_brightness,
                contrast=self._aug_cfg.color_jitter_contrast,
                saturation=self._aug_cfg.color_jitter_saturation,
                hue=self._aug_cfg.color_jitter_hue,
            )
            _, brightness_f, contrast_f, saturation_f, hue_f = T.ColorJitter.get_params(
                cj.brightness, cj.contrast, cj.saturation, cj.hue  # type: ignore[arg-type]
            )
            cj_params = (brightness_f, contrast_f, saturation_f, hue_f)

        apply_noise = self._is_train and self._aug_cfg.random_noise and self._aug_cfg.noise_std > 0

        # --- Apply to every frame ---
        processed: list[Tensor] = []
        for t in range(T_len):
            frame = torch.from_numpy(frames[t]).permute(2, 0, 1)  # (3, H, W) uint8
            if self._needs_resize:
                frame = self._resize(frame)
            frame = _crop(frame)
            if do_flip:
                frame = TF.hflip(frame)
            frame = frame.float() / 255.0
            if cj_params is not None:
                bf, cf, sf, hf = cj_params
                frame = TF.adjust_brightness(frame, bf)
                frame = TF.adjust_contrast(frame, cf)
                frame = TF.adjust_saturation(frame, sf)
                frame = TF.adjust_hue(frame, hf)
            if apply_noise:
                frame = (frame + torch.randn_like(frame) * self._aug_cfg.noise_std).clamp(0.0, 1.0)
            frame = self._normalise(frame)
            processed.append(frame)

        return torch.stack(processed, dim=1)  # (3, T, H, W)


# =============================================================================
# Live-feed preprocessing helper (used by inference)
# =============================================================================


def preprocess_live_frames(
    frames: np.ndarray,
    cfg: object,
    device: torch.device | None = None,
) -> Tensor:
    """Preprocess a raw frame array from a live feed or .mp4 file for inference.

    Applies the same deterministic val-mode pipeline as the Dataset
    (``is_train=False``) so model inputs are consistent between training
    and deployment.

    Handles arbitrary input resolutions by resizing the shorter side to
    ``cfg.dataset.resize_size`` before centre-cropping to
    ``cfg.dataset.frame_size``.  If the input already has the correct
    spatial dimensions the resize step is a no-op.

    Args:
        frames: Raw frames ``(T, H, W, 3)`` uint8 numpy array.  ``T`` may
            differ from ``cfg.dataset.num_frames`` — frames are resampled
            uniformly to match.
        cfg: Fully populated :class:`~config.base_config.Config`.
        device: Target device for the output tensor.

    Returns:
        Float32 tensor ``(1, 3, num_frames, frame_size, frame_size)`` ready
        for :meth:`~models.sign_model.SignModel.predict_topk`.
    """
    ds_cfg = cfg.dataset
    T_in = frames.shape[0]
    T_out = ds_cfg.num_frames

    # Resample to exactly num_frames
    if T_in != T_out:
        indices = np.linspace(0, T_in - 1, num=T_out, dtype=int)
        frames = frames[indices]

    augmentor = VideoAugmentor(
        aug_cfg=cfg.augmentation,
        resize_size=ds_cfg.resize_size,
        crop_size=ds_cfg.frame_size,
        mean=list(ds_cfg.mean),
        std=list(ds_cfg.std),
        is_train=False,
    )
    video = augmentor(frames)  # (3, T, H, W)
    video = video.unsqueeze(0)  # (1, 3, T, H, W)
    if device is not None:
        video = video.to(device)
    return video


# =============================================================================
# WLASL300Dataset
# =============================================================================


class WLASL300Dataset(Dataset):
    """PyTorch Dataset for WLASL300 using pre-extracted JPG frames.

    Loads 16-frame clips from ``preprocessing/<split>/frames/<class_idx>/<video_id>/``,
    applies spatial augmentation, and returns
    ``(video_tensor, word2vec_embedding, label_idx)`` triples.

    Args:
        annotations_file: Path to ``annotations.json``.
        embeddings_file: Path to ``word2vec_embeddings.npy``.
        vocab_file: Path to ``vocab.json``.
        split: ``"train"``, ``"val"``, or ``"test"``.
        cfg: Fully populated :class:`~config.base_config.Config`.
        cache_dir: Optional directory for caching preprocessed ``.pt``
            tensors (val/test only — training always reloads for augmentation
            diversity).

    Raises:
        FileNotFoundError: If any annotation file is missing.
        ValueError: If ``split`` is not recognised.

    Example::

        cfg = Config.from_yaml("config/config.yaml")
        train_ds = WLASL300Dataset(
            annotations_file=cfg.paths.annotations_file,
            embeddings_file=cfg.paths.embeddings_file,
            vocab_file=cfg.paths.vocab_file,
            split="train",
            cfg=cfg,
        )
        video, embedding, label_idx = train_ds[0]
        # video:     torch.Size([3, 16, 256, 256])
        # embedding: torch.Size([300])
        # label_idx: 0
    """

    def __init__(
        self,
        annotations_file: str,
        embeddings_file: str,
        vocab_file: str,
        split: str,
        cfg: object,
        cache_dir: str | None = None,
    ) -> None:
        """Initialise by loading annotation and embedding files.

        Args:
            annotations_file: Path to ``annotations.json``.
            embeddings_file: Path to ``word2vec_embeddings.npy``.
            vocab_file: Path to ``vocab.json``.
            split: Data split — ``"train"``, ``"val"``, or ``"test"``.
            cfg: Top-level config.
            cache_dir: Optional tensor cache directory.

        Raises:
            FileNotFoundError: If a required file is missing.
            ValueError: If ``split`` is not valid.
        """
        if split not in _VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Must be one of {_VALID_SPLITS}.")

        self._split = split
        self._cfg = cfg
        self._dataset_cfg = cfg.dataset
        self._aug_cfg = cfg.augmentation
        self._is_train = split == "train"
        self._cache_dir = Path(cache_dir) if cache_dir else None

        for fpath in (annotations_file, embeddings_file, vocab_file):
            if not Path(fpath).exists():
                raise FileNotFoundError(
                    f"Required annotation file not found: {fpath}\n"
                    "Run dataset/annotations/build_annotations.py first."
                )

        with open(annotations_file, encoding="utf-8") as f:
            all_records: list[dict] = json.load(f)

        with open(vocab_file, encoding="utf-8") as f:
            raw_vocab = json.load(f)
            self._vocab: list[str] = (
                raw_vocab if isinstance(raw_vocab, list) else raw_vocab.get("words", [])
            )

        emb_np = np.load(embeddings_file).astype(np.float32)
        self._embeddings: Tensor = torch.from_numpy(emb_np)

        self._records: list[dict] = [r for r in all_records if r["split"] == split]

        if not self._records:
            log.warning(
                "No records for split '%s' in %s — check build_annotations.py.",
                split,
                annotations_file,
            )

        # Build VideoAugmentor once; stochastic params re-sampled per __getitem__
        self._augmentor = VideoAugmentor(
            aug_cfg=cfg.augmentation,
            resize_size=cfg.dataset.resize_size,
            crop_size=cfg.dataset.frame_size,
            mean=list(cfg.dataset.mean),
            std=list(cfg.dataset.std),
            is_train=self._is_train,
        )

        log.info(
            "WLASL300Dataset  split='%s'  samples=%d  classes=%d",
            split,
            len(self._records),
            len(self._vocab),
        )

    # ---------------------------------------------------------------------- #
    # Dataset protocol
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Return the number of clips in this split.

        Returns:
            Integer sample count.
        """
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        """Load and return one ``(video, embedding, label_idx)`` sample.

        For val/test splits: checks the tensor cache first and writes to it
        on a cache miss.  Training clips are always loaded fresh to preserve
        augmentation diversity.

        Args:
            idx: Index in ``[0, len(self))``.

        Returns:
            - ``video``:     ``(3, T, H, W)`` float32 ImageNet-normalised tensor.
            - ``embedding``: ``(D,)``         float32 L2-normalised Word2Vec vector.
            - ``label_idx``: int class index aligned to ``vocab.json``.

        Raises:
            RuntimeError: If a frame file cannot be opened.
        """
        record = self._records[idx]
        label_idx: int = record["label_idx"]
        embedding: Tensor = self._embeddings[label_idx]

        # --- Cache check (val/test only) ---
        cache_path: Path | None = None
        if self._cache_dir and not self._is_train:
            cache_path = self._cache_dir / f"{record['video_id']}.pt"
            if cache_path.exists():
                return torch.load(cache_path, weights_only=True), embedding, label_idx

        # --- Load JPG frames ---
        ds_cfg = self._dataset_cfg
        num_frames = ds_cfg.num_frames

        # Temporal jitter: shift which frames are used (train only)
        if self._is_train and self._aug_cfg.temporal_jitter:
            jitter = random.randint(
                -self._aug_cfg.temporal_jitter_frames,
                self._aug_cfg.temporal_jitter_frames,
            )
        else:
            jitter = 0

        try:
            frames = _load_jpg_frames(
                frames_dir=record["frames_dir"],
                frame_pattern=record["frame_pattern"],
                num_frames=num_frames,
            )
        except RuntimeError as exc:
            log.error("Failed to load frames for %s: %s", record["video_id"], exc)
            # Return a zero tensor so one bad clip does not crash the epoch
            return (
                torch.zeros(3, num_frames, ds_cfg.frame_size, ds_cfg.frame_size),
                embedding,
                label_idx,
            )

        # Temporal jitter: roll frame order by offset (wrap-around)
        if jitter != 0:
            frames = np.roll(frames, jitter, axis=0)

        # Speed perturbation: resample frame array to simulate playback speed
        if self._is_train and self._aug_cfg.speed_perturbation:
            speed = random.uniform(self._aug_cfg.speed_min, self._aug_cfg.speed_max)
            num_source = max(1, int(round(num_frames / speed)))
            num_source = min(num_source, num_frames)
            src_idx = np.linspace(0, num_frames - 1, num=num_source, dtype=int)
            src_frames = frames[src_idx]
            out_idx = np.linspace(0, num_source - 1, num=num_frames, dtype=int)
            frames = src_frames[out_idx]

        # --- Spatial augmentation ---
        video = self._augmentor(frames)  # (3, T, H, W)

        # --- Cache write (val/test only) ---
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(video, cache_path)

        return video, embedding, label_idx

    # ---------------------------------------------------------------------- #
    # Convenience properties
    # ---------------------------------------------------------------------- #

    @property
    def vocab(self) -> list[str]:
        """Ordered class name list.

        Returns:
            List of class label strings.
        """
        return self._vocab

    @property
    def num_classes(self) -> int:
        """Number of sign language classes.

        Returns:
            Integer class count.
        """
        return len(self._vocab)

    @property
    def embedding_dim(self) -> int:
        """Word2Vec embedding dimensionality.

        Returns:
            Integer dimension (300 for Google News).
        """
        return self._embeddings.shape[1]

    @property
    def class_embedding_matrix(self) -> Tensor:
        """Full ``(num_classes, embedding_dim)`` embedding matrix.

        Returns:
            Float32 tensor for nearest-neighbour inference.
        """
        return self._embeddings

    def get_class_weights(self) -> Tensor:
        """Inverse-frequency class weights for WeightedRandomSampler.

        Returns:
            Float32 tensor of length ``len(self)``.
        """
        from collections import Counter

        counts = Counter(r["label_idx"] for r in self._records)
        weights = [1.0 / counts[r["label_idx"]] for r in self._records]
        return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# DataLoader factory
# =============================================================================


def build_dataloaders(
    cfg: object,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train, val, and test DataLoaders from a Config object.

    When ``cfg.dataset.class_balanced_sampling`` is ``True`` (the default),
    the training loader uses ``WeightedRandomSampler`` so every class
    contributes equally regardless of clip count.  Val and test loaders
    always use sequential sampling.

    Args:
        cfg: Fully populated :class:`~config.base_config.Config`.

    Returns:
        ``(train_loader, val_loader, test_loader)`` tuple.

    Raises:
        FileNotFoundError: If annotation files are missing.

    Example::

        cfg = Config.from_yaml("config/config.yaml")
        train_loader, val_loader, test_loader = build_dataloaders(cfg)
        for videos, embeddings, labels in train_loader:
            # videos: (B, 3, 16, 256, 256)
            ...
    """
    from collections import Counter

    from torch.utils.data import WeightedRandomSampler

    paths = cfg.paths
    shared = dict(
        annotations_file=paths.annotations_file,
        embeddings_file=paths.embeddings_file,
        vocab_file=paths.vocab_file,
        cfg=cfg,
    )

    train_ds = WLASL300Dataset(split="train", cache_dir=None, **shared)
    val_ds = WLASL300Dataset(split="val", cache_dir=paths.processed_dir, **shared)
    test_ds = WLASL300Dataset(split="test", cache_dir=paths.processed_dir, **shared)

    use_balanced = getattr(cfg.dataset, "class_balanced_sampling", True)

    if use_balanced:
        weights = train_ds.get_class_weights()
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        counts = Counter(r["label_idx"] for r in train_ds._records)
        log.info(
            "WeightedRandomSampler enabled — %d classes  " "min=%d  max=%d  mean=%.1f clips/class",
            len(counts),
            min(counts.values()),
            max(counts.values()),
            sum(counts.values()) / len(counts),
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True
        log.info("WeightedRandomSampler disabled — using sequential shuffle")

    loader_kwargs = dict(
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.num_workers > 0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        shuffle=train_shuffle if train_sampler is None else False,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    log.info(
        "DataLoaders — train=%d  val=%d  test=%d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    return train_loader, val_loader, test_loader
