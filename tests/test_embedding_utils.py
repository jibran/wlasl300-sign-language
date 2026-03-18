"""Unit tests for utils/embedding_utils.py.

Tests cover:
- :func:`~utils.embedding_utils.load_vocab` parsing variants.
- :func:`~utils.embedding_utils.load_embeddings` shape and dtype.
- :func:`~utils.embedding_utils.nearest_neighbour` retrieval correctness.
- :func:`~utils.embedding_utils.top1_accuracy` and
  :func:`~utils.embedding_utils.topk_accuracy` boundary cases.
- :func:`~utils.embedding_utils.mean_pairwise_similarity` range.
- :func:`~utils.embedding_utils.find_confusable_pairs` threshold filtering.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_vocab(tmp_path):
    """Write a 6-word vocab.json and return (path, vocab list)."""
    words = ["apple", "banana", "cherry", "date", "elderberry", "fig"]
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(words))
    return p, words


@pytest.fixture
def unit_embeddings():
    """Return a (6, 8) L2-normalised float32 embedding matrix."""
    torch.manual_seed(0)
    emb = F.normalize(torch.randn(6, 8), dim=-1)
    return emb


# =============================================================================
# load_vocab
# =============================================================================


class TestLoadVocab:
    def test_plain_list(self, tmp_path):
        from utils.embedding_utils import load_vocab

        p = tmp_path / "vocab.json"
        p.write_text(json.dumps(["a", "b", "c"]))
        assert load_vocab(p) == ["a", "b", "c"]

    def test_dict_with_words_key(self, tmp_path):
        from utils.embedding_utils import load_vocab

        p = tmp_path / "vocab.json"
        p.write_text(json.dumps({"words": ["x", "y"]}))
        assert load_vocab(p) == ["x", "y"]

    def test_missing_file_raises(self, tmp_path):
        from utils.embedding_utils import load_vocab

        with pytest.raises(FileNotFoundError):
            load_vocab(tmp_path / "nonexistent.json")

    def test_invalid_dict_raises(self, tmp_path):
        from utils.embedding_utils import load_vocab

        p = tmp_path / "vocab.json"
        p.write_text(json.dumps({"labels": ["a"]}))
        with pytest.raises(ValueError, match="words"):
            load_vocab(p)

    def test_invalid_type_raises(self, tmp_path):
        from utils.embedding_utils import load_vocab

        p = tmp_path / "vocab.json"
        p.write_text(json.dumps("not_a_list"))
        with pytest.raises(ValueError):
            load_vocab(p)


# =============================================================================
# load_embeddings
# =============================================================================


class TestLoadEmbeddings:
    def test_returns_tensor_by_default(self, tmp_path, unit_embeddings):
        from utils.embedding_utils import load_embeddings

        p = tmp_path / "emb.npy"
        np.save(str(p), unit_embeddings.numpy())
        result = load_embeddings(p)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (6, 8)
        assert result.dtype == torch.float32

    def test_returns_numpy_when_requested(self, tmp_path, unit_embeddings):
        from utils.embedding_utils import load_embeddings

        p = tmp_path / "emb.npy"
        np.save(str(p), unit_embeddings.numpy())
        result = load_embeddings(p, as_tensor=False)
        assert isinstance(result, np.ndarray)

    def test_missing_file_raises(self, tmp_path):
        from utils.embedding_utils import load_embeddings

        with pytest.raises(FileNotFoundError):
            load_embeddings(tmp_path / "missing.npy")


# =============================================================================
# nearest_neighbour
# =============================================================================


class TestNearestNeighbour:
    def test_top1_returns_itself_when_pred_equals_class(self, unit_embeddings):
        from utils.embedding_utils import nearest_neighbour

        vocab = [str(i) for i in range(6)]
        # Query with class 2 embedding exactly — should rank first
        pred = unit_embeddings[2].unsqueeze(0)
        labels, scores = nearest_neighbour(pred, unit_embeddings, vocab, k=1)
        assert labels[0][0] == "2"
        assert scores[0, 0].item() == pytest.approx(1.0, abs=1e-5)

    def test_returns_k_labels_per_sample(self, unit_embeddings):
        from utils.embedding_utils import nearest_neighbour

        vocab = [str(i) for i in range(6)]
        pred = unit_embeddings[:3]
        labels, scores = nearest_neighbour(pred, unit_embeddings, vocab, k=3)
        assert len(labels) == 3
        assert all(len(row) == 3 for row in labels)
        assert scores.shape == (3, 3)

    def test_scores_descending(self, unit_embeddings):
        from utils.embedding_utils import nearest_neighbour

        vocab = [str(i) for i in range(6)]
        pred = unit_embeddings[:1]
        _, scores = nearest_neighbour(pred, unit_embeddings, vocab, k=6)
        s = scores[0].tolist()
        assert s == sorted(s, reverse=True)

    def test_k_exceeds_classes_raises(self, unit_embeddings):
        from utils.embedding_utils import nearest_neighbour

        vocab = [str(i) for i in range(6)]
        with pytest.raises(ValueError, match="k=10"):
            nearest_neighbour(unit_embeddings[:1], unit_embeddings, vocab, k=10)


# =============================================================================
# top1_accuracy / topk_accuracy
# =============================================================================


class TestAccuracyFunctions:
    def _make(self, num_classes: int = 10, dim: int = 32):
        torch.manual_seed(42)
        class_emb = F.normalize(torch.randn(num_classes, dim), dim=-1)
        return class_emb

    def test_perfect_top1(self):
        from utils.embedding_utils import top1_accuracy

        class_emb = self._make()
        # Predictions identical to class embeddings → top1 = 1.0
        pred = class_emb.clone()
        labels = torch.arange(10)
        assert top1_accuracy(pred, class_emb, labels) == pytest.approx(1.0)

    def test_wrong_top1(self):
        from utils.embedding_utils import top1_accuracy

        class_emb = self._make()
        # All predictions point to class 0 → 1/10 correct
        pred = class_emb[0].unsqueeze(0).expand(10, -1)
        labels = torch.arange(10)
        acc = top1_accuracy(pred, class_emb, labels)
        assert acc == pytest.approx(0.1)

    def test_perfect_topk(self):
        from utils.embedding_utils import topk_accuracy

        class_emb = self._make()
        pred = class_emb.clone()
        labels = torch.arange(10)
        assert topk_accuracy(pred, class_emb, labels, k=1) == pytest.approx(1.0)

    def test_topk_ge_top1(self):
        from utils.embedding_utils import top1_accuracy, topk_accuracy

        class_emb = self._make()
        torch.manual_seed(7)
        pred = F.normalize(torch.randn(10, 32), dim=-1)
        labels = torch.arange(10)
        assert topk_accuracy(pred, class_emb, labels, k=5) >= top1_accuracy(pred, class_emb, labels)


# =============================================================================
# mean_pairwise_similarity
# =============================================================================


class TestMeanPairwiseSimilarity:
    def test_identical_embeddings_returns_one(self):
        from utils.embedding_utils import mean_pairwise_similarity

        emb = torch.ones(5, 8)
        emb = F.normalize(emb, dim=-1)
        result = mean_pairwise_similarity(emb)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_range(self):
        from utils.embedding_utils import mean_pairwise_similarity

        torch.manual_seed(0)
        emb = F.normalize(torch.randn(20, 64), dim=-1)
        result = mean_pairwise_similarity(emb)
        assert -1.0 <= result <= 1.0

    def test_accepts_numpy(self):
        from utils.embedding_utils import mean_pairwise_similarity

        emb = np.random.randn(10, 16).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        result = mean_pairwise_similarity(emb)
        assert isinstance(result, float)


# =============================================================================
# find_confusable_pairs
# =============================================================================


class TestFindConfusablePairs:
    def test_returns_pairs_above_threshold(self):
        from utils.embedding_utils import find_confusable_pairs

        # Two identical vectors → similarity 1.0, well above any threshold
        emb = F.normalize(torch.eye(4, 8), dim=-1)
        # Force pair (0,1) to be identical
        emb[1] = emb[0]
        vocab = ["a", "b", "c", "d"]
        pairs = find_confusable_pairs(emb, vocab, threshold=0.9)
        labels = [(p[0], p[1]) for p in pairs]
        assert ("a", "b") in labels

    def test_empty_when_threshold_too_high(self):
        from utils.embedding_utils import find_confusable_pairs

        torch.manual_seed(1)
        emb = F.normalize(torch.randn(10, 64), dim=-1)
        vocab = [str(i) for i in range(10)]
        pairs = find_confusable_pairs(emb, vocab, threshold=2.0)
        assert pairs == []

    def test_sorted_descending(self):
        from utils.embedding_utils import find_confusable_pairs

        torch.manual_seed(2)
        emb = F.normalize(torch.randn(20, 32), dim=-1)
        vocab = [str(i) for i in range(20)]
        pairs = find_confusable_pairs(emb, vocab, threshold=0.0, top_n=10)
        sims = [p[2] for p in pairs]
        assert sims == sorted(sims, reverse=True)

    def test_top_n_respected(self):
        from utils.embedding_utils import find_confusable_pairs

        torch.manual_seed(3)
        emb = F.normalize(torch.randn(20, 32), dim=-1)
        vocab = [str(i) for i in range(20)]
        pairs = find_confusable_pairs(emb, vocab, threshold=0.0, top_n=5)
        assert len(pairs) <= 5
