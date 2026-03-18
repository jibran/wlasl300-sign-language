"""Smoke tests for utils/visualization.py.

Every plot function is called with minimal dummy data and verified to write
a non-empty PNG file.  We do not assert visual correctness — just that the
functions complete without error and produce output.

All tests use matplotlib's non-interactive ``Agg`` backend so no display is
needed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")


# =============================================================================
# Helpers
# =============================================================================


def _dummy_history(n: int = 5) -> list[float]:
    return [1.0 - i * 0.1 for i in range(n)]


# =============================================================================
# plot_loss_curves
# =============================================================================


class TestPlotLossCurves:
    def test_writes_png(self, tmp_path):
        from utils.visualization import plot_loss_curves

        out = plot_loss_curves(
            train_losses=_dummy_history(),
            val_losses=_dummy_history(),
            save_dir=tmp_path,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_train_only(self, tmp_path):
        from utils.visualization import plot_loss_curves

        out = plot_loss_curves(
            train_losses=_dummy_history(),
            val_losses=[],
            save_dir=tmp_path,
        )
        assert out.exists()


# =============================================================================
# plot_accuracy_curves
# =============================================================================


class TestPlotAccuracyCurves:
    def test_writes_png(self, tmp_path):
        from utils.visualization import plot_accuracy_curves

        out = plot_accuracy_curves(
            top1_history=_dummy_history(),
            top5_history=_dummy_history(),
            save_dir=tmp_path,
        )
        assert out.exists()
        assert out.stat().st_size > 0


# =============================================================================
# plot_throughput
# =============================================================================


class TestPlotThroughput:
    def test_writes_png(self, tmp_path):
        from utils.visualization import plot_throughput

        out = plot_throughput(
            throughput_history=[50.0, 52.0, 48.0, 55.0],
            save_dir=tmp_path,
        )
        assert out.exists()

    def test_empty_history_does_not_raise(self, tmp_path):
        from utils.visualization import plot_throughput

        out = plot_throughput(throughput_history=[], save_dir=tmp_path)
        assert out.exists()


# =============================================================================
# plot_per_class_accuracy
# =============================================================================


class TestPlotPerClassAccuracy:
    def test_writes_png(self, tmp_path):
        from utils.visualization import plot_per_class_accuracy

        per_class = {i: float(i) / 10 for i in range(20)}
        vocab = [f"word_{i}" for i in range(20)]
        out = plot_per_class_accuracy(
            per_class_accuracy=per_class,
            vocab=vocab,
            save_dir=tmp_path,
        )
        assert out.exists()
        assert out.stat().st_size > 0


# =============================================================================
# plot_cosine_similarity_distribution
# =============================================================================


class TestPlotCosineDistribution:
    def test_writes_png(self, tmp_path):
        import torch
        import torch.nn.functional as F

        from utils.visualization import plot_cosine_similarity_distribution

        pred = F.normalize(torch.randn(32, 300), dim=-1)
        target = F.normalize(torch.randn(32, 300), dim=-1)
        out = plot_cosine_similarity_distribution(
            pred_embeddings=pred,
            target_embeddings=target,
            save_dir=tmp_path,
        )
        assert out.exists()
        assert out.stat().st_size > 0
