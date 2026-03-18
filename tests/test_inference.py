"""Unit tests for inference/inference.py and inference/live_inference.py.

Tests cover pure functions only — no GPU, no real video files, no trained
checkpoint.  Functions that require the model or real frames are tested via
mocking.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# =============================================================================
# inference.py — _resolve_device
# =============================================================================


class TestResolveDevice:
    def test_cpu(self):
        from inference.inference import _resolve_device

        assert _resolve_device("cpu") == torch.device("cpu")

    def test_auto_returns_device(self):
        from inference.inference import _resolve_device

        d = _resolve_device("auto")
        assert d in (torch.device("cpu"), torch.device("cuda"))

    def test_cuda_string(self):
        from inference.inference import _resolve_device

        assert _resolve_device("cuda") == torch.device("cuda")


# =============================================================================
# inference.py — _sample_indices
# =============================================================================


class TestSampleIndices:
    def test_length_matches_num_frames(self):
        from inference.inference import _sample_indices

        idx = _sample_indices(total=64, num_frames=16, loop_short=False)
        assert len(idx) == 16

    def test_indices_in_range(self):
        from inference.inference import _sample_indices

        idx = _sample_indices(total=100, num_frames=16, loop_short=False)
        assert all(0 <= i < 100 for i in idx)

    def test_loop_short_fills_to_num_frames(self):
        from inference.inference import _sample_indices

        idx = _sample_indices(total=4, num_frames=16, loop_short=True)
        assert len(idx) == 16
        assert all(0 <= i < 4 for i in idx)

    def test_no_loop_pads_with_last(self):
        from inference.inference import _sample_indices

        idx = _sample_indices(total=4, num_frames=16, loop_short=False)
        assert len(idx) == 16
        # The padding frames must be the last valid index (3)
        assert all(i <= 3 for i in idx)

    def test_total_equals_num_frames(self):
        from inference.inference import _sample_indices

        idx = _sample_indices(total=16, num_frames=16, loop_short=False)
        assert len(idx) == 16


# =============================================================================
# inference.py — format_result
# =============================================================================


class TestFormatResult:
    def _make_result(self):
        return {
            "video": "test.mp4",
            "predictions": [
                {"rank": 1, "label": "book", "score": 0.91},
                {"rank": 2, "label": "read", "score": 0.85},
            ],
            "inference_time_ms": 42.0,
        }

    def test_returns_string(self):
        from inference.inference import format_result

        s = format_result(self._make_result(), verbose=True)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_contains_top_label(self):
        from inference.inference import format_result

        s = format_result(self._make_result(), verbose=True)
        assert "book" in s

    def test_error_result_formatted(self):
        from inference.inference import format_result

        result = {"video": "bad.mp4", "error": "decode failed"}
        s = format_result(result, verbose=True)
        assert isinstance(s, str)


# =============================================================================
# inference.py — compute_accuracy_from_results
# =============================================================================


class TestComputeAccuracyFromResults:
    def _results(self):
        return [
            {
                "video": "WLASL300/0/001.mp4",
                "predictions": [
                    {"rank": 1, "label": "about", "score": 0.9},
                    {"rank": 2, "label": "accident", "score": 0.8},
                ],
            },
            {
                "video": "WLASL300/1/002.mp4",
                "predictions": [
                    {"rank": 1, "label": "accident", "score": 0.88},
                    {"rank": 2, "label": "africa", "score": 0.75},
                ],
            },
            {
                "video": "WLASL300/0/003.mp4",
                "predictions": [
                    {"rank": 1, "label": "wrong", "score": 0.7},
                    {"rank": 2, "label": "about", "score": 0.6},
                ],
            },
        ]

    # vocab aligned with folder2label_str.txt: index 0=about, 1=accident
    _VOCAB = ["about", "accident", "africa"]

    def test_top1_accuracy(self):
        from inference.inference import compute_accuracy_from_results

        results = self._results()
        # video WLASL300/<class_idx>/<video_id>.mp4
        # class 0 → about, class 1 → accident  (via vocab)
        # result[0]: pred=about,    true=about    → correct
        # result[1]: pred=accident, true=accident → correct
        # result[2]: pred=wrong,    true=about    → wrong
        acc = compute_accuracy_from_results(results, k=1, vocab=self._VOCAB)
        assert acc == pytest.approx(2 / 3, abs=1e-5)

    def test_topk_ge_top1(self):
        from inference.inference import compute_accuracy_from_results

        results = self._results()
        top1 = compute_accuracy_from_results(results, k=1, vocab=self._VOCAB)
        top2 = compute_accuracy_from_results(results, k=2, vocab=self._VOCAB)
        assert top2 >= top1

    def test_empty_results(self):
        from inference.inference import compute_accuracy_from_results

        assert compute_accuracy_from_results([], k=1) == pytest.approx(0.0)


# =============================================================================
# live_inference.py — FrameBuffer
# =============================================================================


class TestFrameBuffer:
    def _frame(self, h=256, w=256):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_not_full_when_empty(self):
        from inference.live_inference import FrameBuffer

        buf = FrameBuffer(capacity=16)
        assert not buf.full()
        assert len(buf) == 0

    def test_full_when_at_capacity(self):
        from inference.live_inference import FrameBuffer

        buf = FrameBuffer(capacity=4)
        for _ in range(4):
            buf.push(self._frame())
        assert buf.full()

    def test_slides_on_overflow(self):
        from inference.live_inference import FrameBuffer

        buf = FrameBuffer(capacity=4)
        for i in range(6):
            buf.push(np.full((4, 4, 3), i, dtype=np.uint8))
        # Buffer should hold the last 4 frames (values 2,3,4,5)
        arr = buf.as_array()
        assert arr.shape == (4, 4, 4, 3)
        assert int(arr[0, 0, 0, 0]) == 2

    def test_as_array_shape(self):
        from inference.live_inference import FrameBuffer

        buf = FrameBuffer(capacity=8)
        for _ in range(8):
            buf.push(self._frame(64, 64))
        arr = buf.as_array()
        assert arr.shape == (8, 64, 64, 3)

    def test_reset_clears(self):
        from inference.live_inference import FrameBuffer

        buf = FrameBuffer(capacity=4)
        for _ in range(4):
            buf.push(self._frame())
        buf.reset()
        assert not buf.full()
        assert len(buf) == 0


# =============================================================================
# train/train.py — pure utility functions
# =============================================================================


class TestTrainUtils:
    def test_set_seed_deterministic(self):
        from train.train import set_seed

        set_seed(42)
        a = torch.randn(4)
        set_seed(42)
        b = torch.randn(4)
        assert torch.allclose(a, b)

    def test_get_device_returns_device(self):
        from train.train import get_device

        d = get_device()
        assert isinstance(d, torch.device)
        assert d.type in ("cpu", "cuda")

    def test_get_phase_boundaries(self, cfg):
        from train.train import _get_phase

        # PhaseConfig stores epoch *count*, not absolute end epoch.
        # _get_phase uses 0-based epochs: phase1 covers [0, phase1.epochs),
        # phase2 covers [phase1.epochs, phase1.epochs + phase2.epochs).
        p1_end = cfg.training.phase1.epochs  # last epoch still in phase 1
        p2_end = p1_end + cfg.training.phase2.epochs  # last epoch still in phase 2

        assert _get_phase(0, cfg) == 1  # first epoch → phase 1
        assert _get_phase(p1_end - 1, cfg) == 1  # last epoch of phase 1
        assert _get_phase(p1_end, cfg) == 2  # first epoch of phase 2
        assert _get_phase(p2_end, cfg) == 3  # first epoch of phase 3

    def test_metric_attr_top1(self):
        from train.train import _metric_attr

        assert _metric_attr("top1") == "top1"

    def test_metric_attr_loss(self):
        from train.train import _metric_attr

        assert _metric_attr("loss") == "loss"

    def test_linear_warmup_scales_lr(self, cfg):
        # Build a tiny model via cfg defaults — use projection_head only
        from models.projection_head import ProjectionHead
        from models.sign_model import SignModel
        from train.train import build_optimiser, linear_warmup

        class _Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._output_dim = cfg.model.backbone_output_dim
                self._model_name = "stub"

            @property
            def output_dim(self):
                return self._output_dim

            @property
            def model_name(self):
                return self._model_name

            def forward(self, x):
                return torch.zeros(x.shape[0], self._output_dim)

            def freeze(self):
                pass

            def unfreeze_all(self):
                pass

            def unfreeze_last_n_blocks(self, n):
                pass

            def set_train_mode(self, t):
                pass

            def count_parameters(self, trainable_only=True):
                return 0

        head = ProjectionHead(
            input_dim=cfg.model.backbone_output_dim,
            hidden_dim=cfg.model.projection_hidden_dim,
            output_dim=cfg.model.embedding_dim,
            dropout=0.0,
            l2_normalize=True,
        )
        model = SignModel(backbone=_Stub(), head=head)
        optimiser = build_optimiser(model, cfg, phase=1)
        base_lrs = [g["lr"] for g in optimiser.param_groups]

        # At step=0 warmup_steps=10 → scale = 1/10
        linear_warmup(optimiser, step=0, warmup_steps=10, base_lrs=base_lrs)
        scaled = [g["lr"] for g in optimiser.param_groups]
        for base, s in zip(base_lrs, scaled, strict=True):
            assert s == pytest.approx(base * (1 / 10), rel=1e-4)
