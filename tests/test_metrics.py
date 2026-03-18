"""Unit tests for utils/metrics.py.

Tests cover:
- :class:`~utils.metrics.EpochMetrics` construction and serialisation.
- :class:`~utils.metrics.MetricTracker` accumulation, reset, and compute.
- :func:`~utils.metrics.compute_topk_accuracy` correctness.
- :func:`~utils.metrics.mean_cosine_similarity` range.
- :func:`~utils.metrics.throughput` calculation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

# =============================================================================
# EpochMetrics
# =============================================================================


class TestEpochMetrics:
    def test_to_dict_contains_required_keys(self):
        from utils.metrics import EpochMetrics

        m = EpochMetrics(
            loss=0.5,
            cosine_loss=0.4,
            triplet_loss=0.1,
            top1=0.3,
            top5=0.6,
            mean_cosine_sim=0.7,
            num_samples=128,
            throughput_clips_per_sec=50.0,
        )
        d = m.to_dict()
        for key in ("loss", "top1", "top5", "mean_cosine_sim", "num_samples"):
            assert key in d

    def test_str_is_non_empty(self):
        from utils.metrics import EpochMetrics

        m = EpochMetrics(
            loss=1.0,
            cosine_loss=0.9,
            triplet_loss=0.1,
            top1=0.0,
            top5=0.0,
            mean_cosine_sim=0.0,
            num_samples=0,
            throughput_clips_per_sec=0.0,
        )
        assert len(str(m)) > 0

    def test_values_stored_correctly(self):
        from utils.metrics import EpochMetrics

        m = EpochMetrics(
            loss=0.25,
            cosine_loss=0.2,
            triplet_loss=0.05,
            top1=0.8,
            top5=0.95,
            mean_cosine_sim=0.6,
            num_samples=256,
            throughput_clips_per_sec=100.0,
        )
        assert m.top1 == pytest.approx(0.8)
        assert m.top5 == pytest.approx(0.95)
        assert m.num_samples == 256


# =============================================================================
# MetricTracker
# =============================================================================


class TestMetricTracker:
    def _batch(self, num_classes=10, dim=32, batch_size=8):
        torch.manual_seed(0)
        class_emb = F.normalize(torch.randn(num_classes, dim), dim=-1)
        pred = F.normalize(torch.randn(batch_size, dim), dim=-1)
        labels = torch.randint(0, num_classes, (batch_size,))
        target = class_emb[labels]
        return pred, class_emb, labels, target

    def test_compute_returns_expected_keys(self):
        from utils.metrics import MetricTracker

        tracker = MetricTracker(num_classes=10, topk=(1, 5))
        pred, class_emb, labels, _ = self._batch()
        tracker.update(pred, class_emb, labels)
        result = tracker.compute()
        for key in ("top1", "top5", "mean_cosine_sim"):
            assert key in result

    def test_top1_range(self):
        from utils.metrics import MetricTracker

        tracker = MetricTracker(num_classes=10, topk=(1,))
        pred, class_emb, labels, _ = self._batch()
        tracker.update(pred, class_emb, labels)
        result = tracker.compute()
        assert 0.0 <= result["top1"] <= 1.0

    def test_perfect_top1(self):
        from utils.metrics import MetricTracker

        tracker = MetricTracker(num_classes=10, topk=(1,))
        class_emb = F.normalize(torch.eye(10, 32), dim=-1)
        labels = torch.arange(10)
        pred = class_emb.clone()
        tracker.update(pred, class_emb, labels)
        result = tracker.compute()
        assert result["top1"] == pytest.approx(1.0, abs=1e-5)

    def test_reset_clears_state(self):
        from utils.metrics import MetricTracker

        tracker = MetricTracker(num_classes=10, topk=(1,))
        pred, class_emb, labels, _ = self._batch()
        tracker.update(pred, class_emb, labels)
        tracker.reset()
        with pytest.raises(RuntimeError, match="no accumulated data"):
            tracker.compute()

    def test_num_samples_accumulates(self):
        from utils.metrics import MetricTracker

        tracker = MetricTracker(num_classes=10, topk=(1,))
        pred, class_emb, labels, _ = self._batch(batch_size=8)
        tracker.update(pred, class_emb, labels)
        tracker.update(pred, class_emb, labels)
        assert tracker.num_samples == 16

    def test_loss_dict_averaged(self):
        from utils.metrics import MetricTracker

        tracker = MetricTracker(num_classes=10, topk=(1,))
        pred, class_emb, labels, _ = self._batch()
        loss = {"total_loss": 0.4, "cosine_loss": 0.3, "triplet_loss": 0.1}
        tracker.update(pred, class_emb, labels, loss)
        tracker.update(pred, class_emb, labels, loss)
        result = tracker.compute()
        assert result["loss"] == pytest.approx(0.4, abs=1e-5)

    def test_topk_ge_top1(self):
        from utils.metrics import MetricTracker

        tracker = MetricTracker(num_classes=10, topk=(1, 5))
        pred, class_emb, labels, _ = self._batch()
        tracker.update(pred, class_emb, labels)
        result = tracker.compute()
        assert result["top5"] >= result["top1"] - 1e-6


# =============================================================================
# compute_topk_accuracy
# =============================================================================


class TestComputeTopkAccuracy:
    def test_k1_perfect(self):
        from utils.metrics import compute_topk_accuracy

        C, D = 8, 16
        class_emb = F.normalize(torch.eye(C, D), dim=-1)
        pred = class_emb.clone()
        labels = torch.arange(C)
        assert compute_topk_accuracy(pred, class_emb, labels, k=1) == pytest.approx(1.0)

    def test_k1_all_wrong(self):
        from utils.metrics import compute_topk_accuracy

        C, D = 8, 16
        class_emb = F.normalize(torch.eye(C, D), dim=-1)
        pred = class_emb[0].unsqueeze(0).expand(C, -1)
        labels = torch.arange(C)
        acc = compute_topk_accuracy(pred, class_emb, labels, k=1)
        assert acc == pytest.approx(1.0 / C, abs=1e-5)

    def test_k5_ge_k1(self):
        from utils.metrics import compute_topk_accuracy

        torch.manual_seed(99)
        C, D = 10, 32
        class_emb = F.normalize(torch.randn(C, D), dim=-1)
        pred = F.normalize(torch.randn(C, D), dim=-1)
        labels = torch.arange(C)
        k1 = compute_topk_accuracy(pred, class_emb, labels, k=1)
        k5 = compute_topk_accuracy(pred, class_emb, labels, k=5)
        assert k5 >= k1 - 1e-6


# =============================================================================
# mean_cosine_similarity
# =============================================================================


class TestMeanCosineSimilarity:
    def test_identical_gives_one(self):
        from utils.metrics import mean_cosine_similarity

        v = F.normalize(torch.randn(8, 16), dim=-1)
        assert mean_cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_antipodal_gives_minus_one(self):
        from utils.metrics import mean_cosine_similarity

        v = F.normalize(torch.randn(8, 16), dim=-1)
        assert mean_cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-5)

    def test_range(self):
        from utils.metrics import mean_cosine_similarity

        torch.manual_seed(5)
        a = F.normalize(torch.randn(16, 32), dim=-1)
        b = F.normalize(torch.randn(16, 32), dim=-1)
        result = mean_cosine_similarity(a, b)
        assert -1.0 <= result <= 1.0


# =============================================================================
# throughput
# =============================================================================


class TestThroughput:
    def test_basic(self):
        from utils.metrics import throughput

        assert throughput(100, 2.0) == pytest.approx(50.0)

    def test_zero_elapsed_returns_zero(self):
        from utils.metrics import throughput

        assert throughput(100, 0.0) == pytest.approx(0.0)

    def test_proportional(self):
        from utils.metrics import throughput

        assert throughput(200, 4.0) == pytest.approx(throughput(100, 2.0))
