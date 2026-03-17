"""Utility package for WLASL300 Sign Language Recognition.

Provides helpers for video processing, embedding operations, augmentation,
metrics accumulation, and visualization::

    from utils.video_utils import probe_video, audit_video_dir
    from utils.embedding_utils import load_embeddings, load_vocab, nearest_neighbour
    from utils.augmentation import VideoAugmentorFactory, apply_speed_perturbation
    from utils.metrics import MetricTracker, EpochMetrics
    from utils.visualization import plot_loss_curves, plot_accuracy_curves
"""
