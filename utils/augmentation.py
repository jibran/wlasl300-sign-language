"""Standalone video augmentation helpers for WLASL300 training.

This module provides augmentation utilities that can be used independently
of :class:`~data.dataset.wlasl_dataset.WLASL300Dataset` — for example,
in custom training loops, Colab notebooks, or unit tests.

:class:`VideoAugmentorFactory` builds a configured
:class:`~data.dataset.wlasl_dataset.VideoAugmentor` from a
:class:`~config.base_config.Config` object, eliminating boilerplate.

Temporal augmentation functions (:func:`apply_speed_perturbation`,
:func:`apply_temporal_jitter`) operate on raw frame arrays before they are
converted to tensors, mirroring the logic in
:meth:`~data.dataset.wlasl_dataset.WLASL300Dataset.__getitem__`.

Example::

    from utils.augmentation import VideoAugmentorFactory, apply_speed_perturbation

    cfg = Config.from_yaml("config/config.yaml")
    augmentor = VideoAugmentorFactory.build(cfg, is_train=True)

    frames = np.random.uint8(np.zeros((64, 256, 256, 3)))
    frames = apply_speed_perturbation(frames, speed=1.2, target_frames=64)
    video_tensor = augmentor(frames)   # (3, 64, 224, 224)
"""

from __future__ import annotations

import random

import numpy as np

# =============================================================================
# Temporal augmentations (operate on raw frame arrays)
# =============================================================================


def apply_speed_perturbation(
    frames: np.ndarray,
    speed: float,
    target_frames: int,
) -> np.ndarray:
    """Resample a frame array to simulate a different playback speed.

    A speed > 1.0 (fast) reads more source frames, compressing motion.
    A speed < 1.0 (slow) reads fewer source frames, stretching motion.
    The output is always resampled back to ``target_frames`` via uniform
    indexing so the model always receives a fixed-size tensor.

    Args:
        frames: Input frame array of shape ``(T, H, W, 3)``, uint8.
        speed: Playback speed multiplier.  Values in ``[0.75, 1.25]``
            are typical.  Must be > 0.
        target_frames: Number of frames in the output array.

    Returns:
        Resampled frame array of shape ``(target_frames, H, W, 3)``, uint8.

    Raises:
        ValueError: If ``speed <= 0`` or ``target_frames < 1``.
    """
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")
    if target_frames < 1:
        raise ValueError(f"target_frames must be ≥ 1, got {target_frames}")

    T = frames.shape[0]
    # Number of source frames to read: more for fast, fewer for slow
    num_source = max(1, int(round(target_frames / speed)))
    num_source = min(num_source, T)  # can't exceed available frames

    # Sample num_source frames uniformly from the available T frames
    source_indices = np.linspace(0, T - 1, num=num_source, dtype=int)
    source_frames = frames[source_indices]  # (num_source, H, W, 3)

    # Resample back to target_frames
    if num_source == target_frames:
        return source_frames

    output_indices = np.linspace(0, num_source - 1, num=target_frames, dtype=int)
    return source_frames[output_indices]


def apply_temporal_jitter(
    frame_start: int,
    frame_end: int,
    total_frames: int,
    jitter_frames: int,
) -> tuple[int, int]:
    """Randomly shift the frame sampling window by up to ±jitter_frames.

    Clamps the result so the window stays within ``[0, total_frames - 1]``.

    Args:
        frame_start: Original start frame (1-indexed, from WLASL JSON).
        frame_end: Original end frame (1-indexed, from WLASL JSON).
            Pass -1 to use the last frame of the video.
        total_frames: Total number of frames in the video.
        jitter_frames: Maximum shift magnitude in frames.

    Returns:
        A ``(new_start, new_end)`` tuple with the shifted and clamped
        frame boundaries (still 1-indexed).
    """
    if jitter_frames <= 0:
        return frame_start, frame_end

    offset = random.randint(-jitter_frames, jitter_frames)
    new_start = max(1, frame_start + offset)

    new_end = -1 if frame_end == -1 else max(new_start, min(frame_end + offset, total_frames))

    return new_start, new_end


def random_speed() -> float:
    """Sample a random playback speed from the default range ``[0.75, 1.25]``.

    Returns:
        Float speed multiplier.
    """
    return random.uniform(0.75, 1.25)


# =============================================================================
# Factory for VideoAugmentor
# =============================================================================


class VideoAugmentorFactory:
    """Factory that constructs a :class:`~data.dataset.wlasl_dataset.VideoAugmentor`.

    Reads all augmentation and dataset parameters from a
    :class:`~config.base_config.Config` object, eliminating the need to
    manually pass each argument when creating augmentors outside the
    dataset class.

    Example::

        cfg = Config.from_yaml("config/config.yaml")
        train_aug = VideoAugmentorFactory.build(cfg, is_train=True)
        val_aug   = VideoAugmentorFactory.build(cfg, is_train=False)
    """

    @staticmethod
    def build(cfg: object, is_train: bool) -> object:
        """Construct a :class:`~data.dataset.wlasl_dataset.VideoAugmentor`.

        Args:
            cfg: Fully populated :class:`~config.base_config.Config`.
                Reads ``cfg.augmentation``, ``cfg.dataset.resize_size``,
                ``cfg.dataset.frame_size``, ``cfg.dataset.mean``, and
                ``cfg.dataset.std``.
            is_train: If ``True``, stochastic augmentations are active.

        Returns:
            A configured :class:`~data.dataset.wlasl_dataset.VideoAugmentor`
            instance.
        """
        # Import here to avoid circular dependency at module load time
        from dataset.data.wlasl_dataset import VideoAugmentor

        return VideoAugmentor(
            aug_cfg=cfg.augmentation,
            resize_size=cfg.dataset.resize_size,
            crop_size=cfg.dataset.frame_size,
            mean=list(cfg.dataset.mean),
            std=list(cfg.dataset.std),
            is_train=is_train,
        )


# =============================================================================
# Augmentation parameter samplers
# =============================================================================


def sample_augmentation_params(aug_cfg: object) -> dict:
    """Sample a complete set of augmentation parameters for one clip.

    Useful for testing or for pre-sampling parameters outside the
    :class:`~data.dataset.wlasl_dataset.VideoAugmentor` when you need
    to log or inspect what augmentations were applied.

    Args:
        aug_cfg: :class:`~config.base_config.AugmentationConfig` instance.

    Returns:
        Dict with sampled parameter values::

            {
                "do_flip": True,
                "speed": 1.13,
                "temporal_offset": -2,
                "color_brightness": 0.87,
                "color_contrast": 1.14,
                "color_saturation": 0.95,
                "color_hue": 0.02,
                "noise_std": 0.01,
            }
    """
    params: dict = {}

    params["do_flip"] = (
        aug_cfg.random_horizontal_flip and random.random() < aug_cfg.horizontal_flip_prob
    )

    params["speed"] = (
        random.uniform(aug_cfg.speed_min, aug_cfg.speed_max) if aug_cfg.speed_perturbation else 1.0
    )

    params["temporal_offset"] = (
        random.randint(-aug_cfg.temporal_jitter_frames, aug_cfg.temporal_jitter_frames)
        if aug_cfg.temporal_jitter
        else 0
    )

    if aug_cfg.color_jitter:

        def _jitter(magnitude: float) -> float:
            """Sample a multiplicative jitter factor around 1.0."""
            return random.uniform(max(0.0, 1.0 - magnitude), 1.0 + magnitude)

        params["color_brightness"] = _jitter(aug_cfg.color_jitter_brightness)
        params["color_contrast"] = _jitter(aug_cfg.color_jitter_contrast)
        params["color_saturation"] = _jitter(aug_cfg.color_jitter_saturation)
        params["color_hue"] = random.uniform(-aug_cfg.color_jitter_hue, aug_cfg.color_jitter_hue)
    else:
        params["color_brightness"] = 1.0
        params["color_contrast"] = 1.0
        params["color_saturation"] = 1.0
        params["color_hue"] = 0.0

    params["noise_std"] = aug_cfg.noise_std if aug_cfg.random_noise else 0.0

    return params
