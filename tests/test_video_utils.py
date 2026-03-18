"""Unit tests for utils/video_utils.py.

Tests cover:
- :class:`~utils.video_utils.VideoInfo` and
  :class:`~utils.video_utils.AuditReport` dataclass defaults.
- :func:`~utils.video_utils.uniform_sample_indices` correctness.
- :func:`~utils.video_utils.loop_pad_indices` correctness.
- :func:`~utils.video_utils.probe_video` graceful failure on bad paths.
"""

from __future__ import annotations

import pytest

# =============================================================================
# VideoInfo / AuditReport dataclasses
# =============================================================================


class TestVideoInfo:
    def test_valid_false_by_default(self):
        from utils.video_utils import VideoInfo

        info = VideoInfo(path="x.mp4", is_valid=False)
        assert not info.is_valid

    def test_valid_info(self):
        from utils.video_utils import VideoInfo

        info = VideoInfo(
            path="clip.mp4",
            is_valid=True,
            num_frames=16,
            fps=30.0,
            width=256,
            height=256,
            duration=0.53,
        )
        assert info.num_frames == 16
        assert info.fps == pytest.approx(30.0)


class TestAuditReport:
    def test_empty_report_defaults(self):
        from utils.video_utils import AuditReport

        report = AuditReport()
        assert report.total == 0
        assert report.valid == 0
        assert report.missing_labels == []


# =============================================================================
# uniform_sample_indices
# =============================================================================


class TestUniformSampleIndices:
    def test_exact_count(self):
        from utils.video_utils import uniform_sample_indices

        idx = uniform_sample_indices(total=64, num_frames=16)
        assert len(idx) == 16

    def test_first_is_zero_last_is_total_minus_one(self):
        from utils.video_utils import uniform_sample_indices

        idx = uniform_sample_indices(total=64, num_frames=16)
        assert idx[0] == 0
        assert idx[-1] == 63

    def test_indices_in_range(self):
        from utils.video_utils import uniform_sample_indices

        idx = uniform_sample_indices(total=100, num_frames=16)
        assert all(0 <= i < 100 for i in idx)

    def test_single_frame(self):
        from utils.video_utils import uniform_sample_indices

        idx = uniform_sample_indices(total=1, num_frames=1)
        assert list(idx) == [0]

    def test_invalid_num_frames_raises(self):
        from utils.video_utils import uniform_sample_indices

        with pytest.raises(ValueError):
            uniform_sample_indices(total=16, num_frames=0)


# =============================================================================
# loop_pad_indices
# =============================================================================


class TestLoopPadIndices:
    def test_output_length(self):
        from utils.video_utils import loop_pad_indices

        idx = loop_pad_indices(total=4, num_frames=16)
        assert len(idx) == 16

    def test_all_indices_in_range(self):
        from utils.video_utils import loop_pad_indices

        idx = loop_pad_indices(total=5, num_frames=20)
        assert all(0 <= i < 5 for i in idx)

    def test_total_equals_num_frames(self):
        from utils.video_utils import loop_pad_indices

        idx = loop_pad_indices(total=16, num_frames=16)
        assert len(idx) == 16

    def test_longer_total_falls_back_to_uniform(self):
        from utils.video_utils import loop_pad_indices

        idx = loop_pad_indices(total=32, num_frames=16)
        assert len(idx) == 16
        assert all(0 <= i < 32 for i in idx)


# =============================================================================
# probe_video — graceful failure (no real video file needed)
# =============================================================================


class TestProbeVideo:
    def test_nonexistent_returns_invalid(self, tmp_path):
        from utils.video_utils import probe_video

        info = probe_video(tmp_path / "ghost.mp4")
        assert not info.is_valid
        assert info.error is not None

    def test_not_a_video_returns_invalid(self, tmp_path):
        from utils.video_utils import probe_video

        p = tmp_path / "text.mp4"
        p.write_bytes(b"not a video file")
        info = probe_video(str(p))
        assert not info.is_valid
        assert info.error is not None
