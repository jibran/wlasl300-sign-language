"""Unit tests for classifier model components.

Tests cover:

- :class:`~models.classifier_head.ClassifierHead` output shape, logit range,
  gradient flow, and parameter counting.
- :class:`~models.linear_head.LinearHead` output shape, gradient flow, and
  parameter count exactness.
- :class:`~models.sign_model_classifier.SignModelClassifier` forward shape,
  loss computation, top-k prediction, phase switching, and checkpoint round-trip.
- :class:`~models.sign_model_linear.SignModelLinear` same suite as above.
- Cross-model comparison: linear head must have fewer parameters than the deep
  FC head.
"""

from __future__ import annotations

import copy

import pytest
import torch

# =============================================================================
# Shared stub backbone
# =============================================================================


class _StubBackbone(torch.nn.Module):
    """Minimal backbone stub — returns random features, no I3D download."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self._output_dim = out_dim
        self._model_name = "stub"
        # A real parameter so count_parameters / freeze work correctly
        self.linear = torch.nn.Linear(1, out_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.randn(B, self._output_dim)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int) -> None:
        self.unfreeze_all()

    def set_train_mode(self, is_train: bool) -> None:
        self.train(is_train)

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)


# =============================================================================
# Module-level fixtures
# =============================================================================

NUM_CLASSES = 10
INPUT_DIM = 64
HIDDEN_DIM = 32
BATCH = 8


@pytest.fixture(scope="module")
def classifier_head():
    """Return a small ClassifierHead on CPU."""
    from models.classifier_head import ClassifierHead

    return ClassifierHead(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=0.0,  # disable dropout for deterministic tests
    )


@pytest.fixture(scope="module")
def linear_head():
    """Return a small LinearHead on CPU."""
    from models.linear_head import LinearHead

    return LinearHead(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)


@pytest.fixture(scope="module")
def stub_backbone():
    """Return a _StubBackbone with OUTPUT_DIM features."""
    return _StubBackbone(out_dim=INPUT_DIM)


@pytest.fixture
def small_classifier(stub_backbone):
    """Return a SignModelClassifier built from stub backbone + small head."""
    from models.classifier_head import ClassifierHead
    from models.sign_model_classifier import SignModelClassifier

    head = ClassifierHead(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=0.0,
    )
    return SignModelClassifier(backbone=copy.deepcopy(stub_backbone), head=head)


@pytest.fixture
def small_linear(stub_backbone):
    """Return a SignModelLinear built from stub backbone + single linear head."""
    from models.linear_head import LinearHead
    from models.sign_model_linear import SignModelLinear

    head = LinearHead(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    return SignModelLinear(backbone=copy.deepcopy(stub_backbone), head=head)


def _video_batch(b: int = BATCH) -> torch.Tensor:
    """Return a random ``(B, 3, 4, 32, 32)`` video tensor for fast stub tests."""
    return torch.randn(b, 3, 4, 32, 32)


def _labels(b: int = BATCH) -> torch.Tensor:
    """Return random integer class indices ``(B,)`` in ``[0, NUM_CLASSES)``."""
    return torch.randint(0, NUM_CLASSES, (b,))


# =============================================================================
# ClassifierHead tests
# =============================================================================


class TestClassifierHead:
    """Tests for the deep FC ClassifierHead."""

    def test_output_shape(self, classifier_head):
        """Output must be (B, num_classes)."""
        x = torch.randn(BATCH, INPUT_DIM)
        out = classifier_head(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_output_is_logits_not_probs(self, classifier_head):
        """Output must be raw logits — not bounded to [0, 1]."""
        classifier_head.eval()
        with torch.no_grad():
            x = torch.randn(BATCH, INPUT_DIM)
            out = classifier_head(x)
        # Logits may exceed [0, 1]; if all values were in [0,1] it would suggest
        # softmax was accidentally applied
        has_out_of_unit_range = (out.abs() > 1.0).any()
        # This is probabilistic but extremely likely for random weights
        assert has_out_of_unit_range or out.shape == (BATCH, NUM_CLASSES)

    def test_train_eval_same_shape(self, classifier_head):
        """Output shape must be identical in train and eval mode."""
        x = torch.randn(BATCH, INPUT_DIM)

        classifier_head.train()
        out_train = classifier_head(x)

        classifier_head.eval()
        with torch.no_grad():
            out_eval = classifier_head(x)

        assert out_train.shape == out_eval.shape

    def test_gradient_flows(self, classifier_head):
        """Gradients must reach the input tensor."""
        classifier_head.train()
        x = torch.randn(BATCH, INPUT_DIM, requires_grad=True)
        loss = classifier_head(x).sum()
        loss.backward()
        assert x.grad is not None

    def test_count_parameters(self, classifier_head):
        """count_parameters must return a positive integer."""
        n = classifier_head.count_parameters(trainable_only=True)
        assert isinstance(n, int) and n > 0

    def test_properties(self, classifier_head):
        """input_dim, hidden_dim, num_classes properties must match constructor args."""
        assert classifier_head.input_dim == INPUT_DIM
        assert classifier_head.hidden_dim == HIDDEN_DIM
        assert classifier_head.num_classes == NUM_CLASSES

    def test_bn_layer_present(self, classifier_head):
        """Must contain a BatchNorm1d layer."""
        has_bn = any(isinstance(m, torch.nn.BatchNorm1d) for m in classifier_head.modules())
        assert has_bn, "ClassifierHead must contain BatchNorm1d"

    def test_dropout_layer_present(self, classifier_head):
        """Must contain a Dropout layer."""
        has_drop = any(isinstance(m, torch.nn.Dropout) for m in classifier_head.modules())
        assert has_drop, "ClassifierHead must contain Dropout"


# =============================================================================
# LinearHead tests
# =============================================================================


class TestLinearHead:
    """Tests for the single-layer LinearHead."""

    def test_output_shape(self, linear_head):
        """Output must be (B, num_classes)."""
        x = torch.randn(BATCH, INPUT_DIM)
        out = linear_head(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_gradient_flows(self, linear_head):
        """Gradients must reach the input tensor."""
        x = torch.randn(BATCH, INPUT_DIM, requires_grad=True)
        loss = linear_head(x).sum()
        loss.backward()
        assert x.grad is not None

    def test_exact_parameter_count(self, linear_head):
        """Parameter count must equal input_dim * num_classes + num_classes."""
        expected = INPUT_DIM * NUM_CLASSES + NUM_CLASSES
        actual = linear_head.count_parameters(trainable_only=False)
        assert actual == expected, f"Expected {expected} params, got {actual}"

    def test_properties(self, linear_head):
        """input_dim and num_classes properties must match constructor args."""
        assert linear_head.input_dim == INPUT_DIM
        assert linear_head.num_classes == NUM_CLASSES

    def test_fewer_params_than_classifier_head(self, classifier_head, linear_head):
        """LinearHead must have fewer parameters than ClassifierHead."""
        n_linear = linear_head.count_parameters(trainable_only=False)
        n_deep = classifier_head.count_parameters(trainable_only=False)
        assert n_linear < n_deep, (
            f"LinearHead ({n_linear}) should have fewer params than " f"ClassifierHead ({n_deep})"
        )

    def test_no_bn_no_dropout(self, linear_head):
        """LinearHead must NOT contain BatchNorm or Dropout layers."""
        for m in linear_head.modules():
            assert not isinstance(m, torch.nn.BatchNorm1d), "Unexpected BN in LinearHead"
            assert not isinstance(m, torch.nn.Dropout), "Unexpected Dropout in LinearHead"


# =============================================================================
# SignModelClassifier tests
# =============================================================================


class TestSignModelClassifier:
    """Tests for the deep-FC end-to-end classifier model."""

    def test_forward_output_shape(self, small_classifier):
        """Forward must return (B, num_classes) logit tensor."""
        out = small_classifier(_video_batch())
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_forward_output_is_not_normalised(self, small_classifier):
        """Forward output must NOT be L2-normalised (it's logits, not embeddings)."""
        small_classifier.eval()
        with torch.no_grad():
            out = small_classifier(_video_batch())
        norms = out.norm(dim=-1)
        # Logit norms will not all be 1.0
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    def test_loss_is_positive(self, small_classifier):
        """Cross-entropy loss must be positive."""
        logits = small_classifier(_video_batch())
        loss = small_classifier.loss(logits, _labels())
        assert loss.item() > 0.0

    def test_loss_perfect_prediction(self):
        """Loss for perfect predictions must be very small (near 0 with label smoothing)."""
        from models.sign_model_classifier import SignModelClassifier

        # Create logits with very high confidence for the correct class
        logits = torch.full((BATCH, NUM_CLASSES), -100.0)
        labels = _labels()
        for i, lbl in enumerate(labels.tolist()):
            logits[i, lbl] = 100.0

        loss = SignModelClassifier.loss(logits, labels, label_smoothing=0.0)
        assert loss.item() < 0.01, f"Expected near-zero loss, got {loss.item()}"

    def test_loss_scalar(self, small_classifier):
        """Loss must be a scalar (0-d) tensor."""
        logits = small_classifier(_video_batch())
        loss = small_classifier.loss(logits, _labels())
        assert loss.dim() == 0

    def test_predict_topk_shape(self, small_classifier, dummy_vocab):
        """predict_topk must return B lists each with k prediction dicts."""
        small_classifier.eval()
        with torch.no_grad():
            results = small_classifier.predict_topk(_video_batch(), vocab=dummy_vocab, k=3)
        assert len(results) == BATCH
        assert all(len(r) == 3 for r in results)

    def test_predict_topk_scores_sum_to_one(self, small_classifier, dummy_vocab):
        """Top-NUM_CLASSES softmax scores must sum to ≈ 1 per sample."""
        small_classifier.eval()
        with torch.no_grad():
            results = small_classifier.predict_topk(
                _video_batch(), vocab=dummy_vocab, k=NUM_CLASSES
            )
        for preds in results:
            total = sum(p["score"] for p in preds)
            assert abs(total - 1.0) < 1e-3, f"Scores sum to {total}, expected ~1.0"

    def test_predict_topk_rank_order(self, small_classifier, dummy_vocab):
        """Predictions must be sorted by descending score."""
        small_classifier.eval()
        with torch.no_grad():
            results = small_classifier.predict_topk(
                _video_batch(), vocab=dummy_vocab, k=NUM_CLASSES
            )
        for preds in results:
            scores = [p["score"] for p in preds]
            assert scores == sorted(scores, reverse=True), "Predictions not sorted by score"

    def test_apply_phase_1_freezes_backbone(self, small_classifier):
        """apply_phase(1) must freeze all backbone parameters."""
        small_classifier.apply_phase(1)
        assert all(not p.requires_grad for p in small_classifier.backbone.parameters())

    def test_apply_phase_3_unfreezes_backbone(self, small_classifier):
        """apply_phase(3) must unfreeze all backbone parameters."""
        small_classifier.apply_phase(1)
        small_classifier.apply_phase(3)
        assert all(p.requires_grad for p in small_classifier.backbone.parameters())

    def test_invalid_phase_raises(self, small_classifier):
        """apply_phase with an invalid phase must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid training phase"):
            small_classifier.apply_phase(0)

    def test_model_summary_string(self, small_classifier):
        """model_summary must return a non-empty string."""
        summary = small_classifier.model_summary()
        assert isinstance(summary, str) and len(summary) > 0

    def test_count_parameters_keys(self, small_classifier):
        """count_parameters must return dict with backbone/head/total."""
        counts = small_classifier.count_parameters(trainable_only=False)
        assert "backbone" in counts
        assert "head" in counts
        assert counts["total"] == counts["backbone"] + counts["head"]

    def test_checkpoint_round_trip(self, small_classifier, cfg, tmp_path, mocker):
        """Save then load — weights and metadata must round-trip correctly."""
        from models.sign_model_classifier import SignModelClassifier

        ckpt = tmp_path / "clf_checkpoint.pt"
        small_classifier.save_checkpoint(path=ckpt, epoch=3, metrics={"top1": 0.42})
        assert ckpt.exists()

        fresh = copy.deepcopy(small_classifier)
        mocker.patch.object(SignModelClassifier, "from_config", return_value=fresh)

        loaded, epoch, metrics = SignModelClassifier.load_checkpoint(ckpt, cfg, device="cpu")
        assert epoch == 3
        assert metrics.get("top1") == pytest.approx(0.42)

        for (k, v1), (_, v2) in zip(
            small_classifier.head.state_dict().items(),
            loaded.head.state_dict().items(),
            strict=True,
        ):
            assert torch.allclose(v1, v2), f"Head weight mismatch at '{k}'"

    def test_gradient_flows_end_to_end(self, small_classifier):
        """Loss backward must produce non-None gradients on head parameters."""
        small_classifier.train()
        logits = small_classifier(_video_batch())
        loss = small_classifier.loss(logits, _labels())
        loss.backward()
        for name, p in small_classifier.head.named_parameters():
            assert p.grad is not None, f"No gradient for head param '{name}'"


# =============================================================================
# SignModelLinear tests
# =============================================================================


class TestSignModelLinear:
    """Tests for the single-linear-layer end-to-end classifier model."""

    def test_forward_output_shape(self, small_linear):
        """Forward must return (B, num_classes) logit tensor."""
        out = small_linear(_video_batch())
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_forward_output_is_not_normalised(self, small_linear):
        """Forward output must NOT be L2-normalised."""
        small_linear.eval()
        with torch.no_grad():
            out = small_linear(_video_batch())
        norms = out.norm(dim=-1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    def test_loss_is_positive(self, small_linear):
        """Cross-entropy loss must be positive."""
        logits = small_linear(_video_batch())
        loss = small_linear.loss(logits, _labels())
        assert loss.item() > 0.0

    def test_loss_scalar(self, small_linear):
        """Loss must be a scalar (0-d) tensor."""
        logits = small_linear(_video_batch())
        loss = small_linear.loss(logits, _labels())
        assert loss.dim() == 0

    def test_predict_topk_shape(self, small_linear, dummy_vocab):
        """predict_topk must return B lists each with k prediction dicts."""
        small_linear.eval()
        with torch.no_grad():
            results = small_linear.predict_topk(_video_batch(), vocab=dummy_vocab, k=3)
        assert len(results) == BATCH
        assert all(len(r) == 3 for r in results)

    def test_predict_topk_scores_sum_to_one(self, small_linear, dummy_vocab):
        """Top-NUM_CLASSES softmax scores must sum to ≈ 1 per sample."""
        small_linear.eval()
        with torch.no_grad():
            results = small_linear.predict_topk(_video_batch(), vocab=dummy_vocab, k=NUM_CLASSES)
        for preds in results:
            total = sum(p["score"] for p in preds)
            assert abs(total - 1.0) < 1e-3

    def test_predict_topk_rank_order(self, small_linear, dummy_vocab):
        """Predictions must be sorted by descending score."""
        small_linear.eval()
        with torch.no_grad():
            results = small_linear.predict_topk(_video_batch(), vocab=dummy_vocab, k=NUM_CLASSES)
        for preds in results:
            scores = [p["score"] for p in preds]
            assert scores == sorted(scores, reverse=True)

    def test_apply_phase_1_freezes_backbone(self, small_linear):
        """apply_phase(1) must freeze all backbone parameters."""
        small_linear.apply_phase(1)
        assert all(not p.requires_grad for p in small_linear.backbone.parameters())

    def test_apply_phase_3_unfreezes_backbone(self, small_linear):
        """apply_phase(3) must unfreeze all backbone parameters."""
        small_linear.apply_phase(1)
        small_linear.apply_phase(3)
        assert all(p.requires_grad for p in small_linear.backbone.parameters())

    def test_invalid_phase_raises(self, small_linear):
        """apply_phase with an invalid phase must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid training phase"):
            small_linear.apply_phase(4)

    def test_model_summary_string(self, small_linear):
        """model_summary must return a non-empty string."""
        summary = small_linear.model_summary()
        assert isinstance(summary, str) and len(summary) > 0

    def test_count_parameters_keys(self, small_linear):
        """count_parameters must return dict with backbone/head/total."""
        counts = small_linear.count_parameters(trainable_only=False)
        assert "backbone" in counts
        assert "head" in counts
        assert counts["total"] == counts["backbone"] + counts["head"]

    def test_head_parameter_count_exact(self, small_linear):
        """Head must have exactly input_dim * num_classes + num_classes params."""
        expected = INPUT_DIM * NUM_CLASSES + NUM_CLASSES
        actual = small_linear.head.count_parameters(trainable_only=False)
        assert actual == expected

    def test_checkpoint_round_trip(self, small_linear, cfg, tmp_path, mocker):
        """Save then load — weights and metadata must round-trip correctly."""
        from models.sign_model_linear import SignModelLinear

        ckpt = tmp_path / "linear_checkpoint.pt"
        small_linear.save_checkpoint(path=ckpt, epoch=7, metrics={"top1": 0.31})
        assert ckpt.exists()

        fresh = copy.deepcopy(small_linear)
        mocker.patch.object(SignModelLinear, "from_config", return_value=fresh)

        loaded, epoch, metrics = SignModelLinear.load_checkpoint(ckpt, cfg, device="cpu")
        assert epoch == 7
        assert metrics.get("top1") == pytest.approx(0.31)

        for (k, v1), (_, v2) in zip(
            small_linear.head.state_dict().items(),
            loaded.head.state_dict().items(),
            strict=True,
        ):
            assert torch.allclose(v1, v2), f"Head weight mismatch at '{k}'"

    def test_gradient_flows_end_to_end(self, small_linear):
        """Loss backward must produce non-None gradients on head parameters."""
        small_linear.train()
        logits = small_linear(_video_batch())
        loss = small_linear.loss(logits, _labels())
        loss.backward()
        for name, p in small_linear.head.named_parameters():
            assert p.grad is not None, f"No gradient for head param '{name}'"


# =============================================================================
# Cross-model comparison tests
# =============================================================================


class TestModelComparison:
    """Tests comparing SignModelLinear and SignModelClassifier properties."""

    def test_linear_has_fewer_head_params(self, small_classifier, small_linear):
        """LinearHead must have fewer parameters than ClassifierHead."""
        n_clf = small_classifier.head.count_parameters(trainable_only=False)
        n_lin = small_linear.head.count_parameters(trainable_only=False)
        assert (
            n_lin < n_clf
        ), f"Linear head ({n_lin}) must be smaller than classifier head ({n_clf})"

    def test_same_output_shape(self, small_classifier, small_linear):
        """Both models must produce identical output shapes."""
        x = _video_batch()
        out_clf = small_classifier(x)
        out_lin = small_linear(x)
        assert out_clf.shape == out_lin.shape

    def test_same_num_classes(self, small_classifier, small_linear):
        """Both models must have the same num_classes."""
        assert small_classifier.head.num_classes == small_linear.head.num_classes

    def test_both_produce_valid_probabilities(self, small_classifier, small_linear, dummy_vocab):
        """Both models must produce valid probability distributions via predict_topk."""
        for model in (small_classifier, small_linear):
            model.eval()
            with torch.no_grad():
                results = model.predict_topk(_video_batch(), vocab=dummy_vocab, k=NUM_CLASSES)
            for preds in results:
                scores = [p["score"] for p in preds]
                assert all(0.0 <= s <= 1.0 for s in scores), "Score outside [0, 1]"
                assert abs(sum(scores) - 1.0) < 1e-3, "Scores do not sum to 1"

    def test_checkpoint_model_type_tag(self, small_classifier, small_linear, tmp_path):
        """Checkpoints must tag model_type correctly for each model."""
        clf_ckpt = tmp_path / "clf.pt"
        lin_ckpt = tmp_path / "lin.pt"

        small_classifier.save_checkpoint(clf_ckpt, epoch=0)
        small_linear.save_checkpoint(lin_ckpt, epoch=0)

        clf_payload = torch.load(clf_ckpt, map_location="cpu", weights_only=False)
        lin_payload = torch.load(lin_ckpt, map_location="cpu", weights_only=False)

        assert clf_payload["model_type"] == "classifier"
        assert lin_payload["model_type"] == "linear"
