"""Inference package for WLASL300 Sign Language Recognition.

Exposes the core prediction functions for use outside the CLI::

    from inference.inference import predict_single, predict_batch

    result = predict_single(
        video_path="path/to/video.mp4",
        model=model,
        class_embeddings=class_embeddings,
        vocab=vocab,
        cfg=cfg,
        device=device,
    )
"""

from inference.inference import (
    compute_accuracy_from_results,
    format_result,
    predict_batch,
    predict_directory,
    predict_single,
    preprocess_video,
)

__all__ = [
    "predict_single",
    "predict_batch",
    "predict_directory",
    "preprocess_video",
    "compute_accuracy_from_results",
    "format_result",
]
