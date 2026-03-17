"""Unit tests for the annotation pipeline.

Tests cover:

- :func:`~data.annotations.build_annotations.load_folder2label` parsing.
- :func:`~data.annotations.build_annotations.build_vocab` correctness.
- :func:`~data.annotations.build_annotations.assign_label_indices` alignment.
- :func:`~data.annotations.build_annotations.build_splits_map` coverage.
- :func:`~data.annotations.build_annotations.write_outputs` file creation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# =============================================================================
# Helpers
# =============================================================================

# label_map mirrors what load_folder2label returns from folder2label_str.txt
_LABEL_MAP: dict[int, str] = {0: "book", 1: "drink", 2: "computer"}


def _make_clips(label_map: dict[int, str] = _LABEL_MAP, clips_per_class: int = 4) -> list[dict]:
    """Return minimal clip records matching the new annotation format."""
    clips = []
    for class_idx, label in label_map.items():
        for i in range(clips_per_class):
            video_id = f"{class_idx:02d}{i:02d}"
            split = "train" if i < 3 else ("val" if i == 3 else "test")
            clips.append(
                {
                    "video_id": video_id,
                    "frames_dir": f"/fake/preprocessing/{split}/frames/{class_idx}/{video_id}",
                    "video_path": f"/fake/WLASL300/{class_idx}/{video_id}.mp4",
                    "label": label,
                    "label_idx": class_idx,
                    "class_idx": class_idx,
                    "split": split,
                    "num_frames": 16,
                    "frame_pattern": f"{class_idx}_{{}}.jpg",
                }
            )
    return clips


# =============================================================================
# load_folder2label tests
# =============================================================================


class TestLoadFolder2Label:
    """Tests for load_folder2label."""

    def test_parses_correctly(self, tmp_path):
        """load_folder2label must return correct int→str mapping."""
        from dataset.annotations.build_annotations import load_folder2label

        f = tmp_path / "folder2label_str.txt"
        f.write_text("0 book\n1 drink\n2 computer\n")
        result = load_folder2label(f)
        assert result == {0: "book", 1: "drink", 2: "computer"}

    def test_missing_file_raises(self, tmp_path):
        """load_folder2label must raise FileNotFoundError for missing file."""
        from dataset.annotations.build_annotations import load_folder2label

        with pytest.raises(FileNotFoundError):
            load_folder2label(tmp_path / "nonexistent.txt")

    def test_bad_line_raises(self, tmp_path):
        """load_folder2label must raise ValueError for unparseable lines."""
        from dataset.annotations.build_annotations import load_folder2label

        f = tmp_path / "bad.txt"
        f.write_text("0\n")  # missing label
        with pytest.raises(ValueError, match="Unparseable"):
            load_folder2label(f)


# =============================================================================
# build_vocab tests
# =============================================================================


class TestBuildVocab:
    """Tests for build_vocab."""

    def test_returns_labels_ordered_by_index(self):
        """vocab[i] must equal the label for class index i."""
        from dataset.annotations.build_annotations import build_vocab

        vocab = build_vocab(_LABEL_MAP)
        for idx, label in _LABEL_MAP.items():
            assert vocab[idx] == label

    def test_correct_length(self):
        """Vocab length must equal the number of classes."""
        from dataset.annotations.build_annotations import build_vocab

        vocab = build_vocab(_LABEL_MAP)
        assert len(vocab) == len(_LABEL_MAP)

    def test_non_contiguous_raises(self):
        """build_vocab must raise ValueError for non-contiguous indices."""
        from dataset.annotations.build_annotations import build_vocab

        with pytest.raises(ValueError, match="missing"):
            build_vocab({0: "book", 2: "computer"})  # index 1 is missing


# =============================================================================
# assign_label_indices tests
# =============================================================================


class TestAssignLabelIndices:
    """Tests for assign_label_indices."""

    def test_indices_added_to_records(self):
        """Each record must have a 'label_idx' int field after assignment."""
        from dataset.annotations.build_annotations import assign_label_indices, build_vocab

        clips = _make_clips()
        vocab = build_vocab(_LABEL_MAP)
        result = assign_label_indices(clips, vocab)
        for r in result:
            assert "label_idx" in r
            assert isinstance(r["label_idx"], int)

    def test_indices_align_with_vocab(self):
        """vocab[label_idx] must equal record['label'] for every record."""
        from dataset.annotations.build_annotations import assign_label_indices, build_vocab

        clips = _make_clips()
        vocab = build_vocab(_LABEL_MAP)
        result = assign_label_indices(clips, vocab)
        for r in result:
            assert vocab[r["label_idx"]] == r["label"]


# =============================================================================
# build_splits_map tests
# =============================================================================


class TestBuildSplitsMap:
    """Tests for build_splits_map."""

    def test_all_video_ids_covered(self):
        """Union of all splits must equal all video IDs."""
        from dataset.annotations.build_annotations import build_splits_map

        clips = _make_clips()
        splits_map = build_splits_map(clips)
        all_ids = {c["video_id"] for c in clips}
        mapped_ids = set(splits_map["train"]) | set(splits_map["val"]) | set(splits_map["test"])
        assert all_ids == mapped_ids

    def test_splits_keys_present(self):
        """splits_map must contain train, val, and test keys."""
        from dataset.annotations.build_annotations import build_splits_map

        splits_map = build_splits_map(_make_clips())
        assert set(splits_map.keys()) == {"train", "val", "test"}


# =============================================================================
# write_outputs tests
# =============================================================================


class TestWriteOutputs:
    """Tests for write_outputs file creation."""

    def _run_write(self, tmp_path):
        from dataset.annotations.build_annotations import (
            assign_label_indices,
            build_splits_map,
            build_vocab,
            write_outputs,
        )

        clips = _make_clips()
        vocab = build_vocab(_LABEL_MAP)
        clips = assign_label_indices(clips, vocab)
        splits_map = build_splits_map(clips)
        embeddings = np.zeros((len(vocab), 300), dtype=np.float32)

        write_outputs(
            out_dir=tmp_path,
            clips=clips,
            vocab=vocab,
            splits_map=splits_map,
            embeddings=embeddings,
            oov_words=[],
            incomplete_clips=[],
        )
        return vocab

    def test_all_output_files_created(self, tmp_path):
        """write_outputs must create all five expected files."""
        self._run_write(tmp_path)
        for fname in (
            "annotations.json",
            "vocab.json",
            "splits.json",
            "word2vec_embeddings.npy",
            "oov_report.txt",
        ):
            assert (tmp_path / fname).exists(), f"Missing output file: {fname}"

    def test_vocab_json_is_plain_list(self, tmp_path):
        """vocab.json written by write_outputs must be a plain JSON array."""
        vocab = self._run_write(tmp_path)
        loaded = json.loads((tmp_path / "vocab.json").read_text())
        assert isinstance(loaded, list)
        assert loaded == vocab

    def test_annotations_json_has_required_fields(self, tmp_path):
        """Every record in annotations.json must have the required keys."""
        self._run_write(tmp_path)
        records = json.loads((tmp_path / "annotations.json").read_text())
        required = {"video_id", "frames_dir", "label", "label_idx", "split"}
        for r in records:
            assert required.issubset(r.keys()), f"Missing keys in record: {r}"
