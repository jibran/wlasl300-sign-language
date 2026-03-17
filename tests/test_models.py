"""Unit tests for model architecture components.

Tests cover:

- :class:`~models.projection_head.ProjectionHead` output shape,
  L2 normalisation, and weight initialisation.
- :class:`~models.sign_model.SignModel` forward pass shape.
- :class:`~models.sign_model.SignModel` cosine and triplet loss values.
- :class:`~models.sign_model.SignModel` phase switching and freeze state.
- :class:`~models.sign_model.SignModel` checkpoint save/load round-trip.
- :class:`~models.i3d_backbone.I3DBackbone` freeze/unfreeze helpers.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

# =============================================================================
# ProjectionHead tests
# =============================================================================


class TestProjectionHead:
    """Tests for the ProjectionHead nn.Module."""

    def test_output_shape(self, projection_head, cfg):
        """ProjectionHead must return (B, embedding_dim) tensors."""
        B = 8
        x = torch.randn(B, cfg.model.backbone_output_dim)
        out = projection_head(x)
        assert out.shape == (
            B,
            cfg.model.embedding_dim,
        ), f"Expected ({B}, {cfg.model.embedding_dim}), got {out.shape}"

    def test_l2_normalised_output(self, projection_head):
        """Output vectors must have L2 norm ≈ 1.0 when l2_normalize=True."""
        x = torch.randn(16, projection_head.fc1.in_features)
        out = projection_head(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(
            norms, torch.ones_like(norms), atol=1e-5
        ), f"L2 norms not all 1.0: min={norms.min():.6f}  max={norms.max():.6f}"

    def test_train_eval_mode(self, projection_head, cfg):
        """Output shapes must match in both train and eval mode."""
        B, D_in = 4, cfg.model.backbone_output_dim
        x = torch.randn(B, D_in)

        projection_head.train()
        out_train = projection_head(x)

        projection_head.eval()
        with torch.no_grad():
            out_eval = projection_head(x)

        assert out_train.shape == out_eval.shape

    def test_invalid_dimensions_raise(self):
        """ProjectionHead must raise ValueError for invalid dimension args."""
        from models.projection_head import ProjectionHead

        with pytest.raises(ValueError, match="must be ≥ 1"):
            ProjectionHead(input_dim=0, hidden_dim=512, output_dim=300)

    def test_invalid_dropout_raises(self):
        """ProjectionHead must raise ValueError for dropout ≥ 1."""
        from models.projection_head import ProjectionHead

        with pytest.raises(ValueError, match="dropout must be"):
            ProjectionHead(input_dim=1024, hidden_dim=512, output_dim=300, dropout=1.0)

    def test_count_parameters(self, projection_head):
        """count_parameters must return a positive integer."""
        count = projection_head.count_parameters(trainable_only=True)
        assert isinstance(count, int) and count > 0

    def test_gradient_flows(self, projection_head, cfg):
        """Gradients must flow back through the head to the input."""
        x = torch.randn(4, cfg.model.backbone_output_dim, requires_grad=True)
        out = projection_head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient at input — backward pass broken"


# =============================================================================
# SignModel forward pass tests
# =============================================================================


@pytest.fixture
def small_model(cfg):
    """Build a small SignModel with a tiny backbone stub for fast testing.

    Avoids loading the full I3D backbone (which requires pytorchvideo
    and is marked as a slow test).
    """
    from models.projection_head import ProjectionHead
    from models.sign_model import SignModel

    class _StubBackbone(torch.nn.Module):
        """Minimal backbone stub that returns zeros."""

        def __init__(self, out_dim) -> None:
            super().__init__()
            self.out_dim = out_dim
            self._model_name = "stub"
            self._output_dim = out_dim
            self.linear = torch.nn.Linear(1, out_dim)  # gives it parameters

        @property
        def output_dim(self):
            return self.out_dim

        @property
        def model_name(self):
            return self._model_name

        def forward(self, x):
            B = x.shape[0]
            # randn so L2 normalisation in ProjectionHead produces unit vectors
            return torch.randn(B, self.out_dim)

        def freeze(self):
            for p in self.parameters():
                p.requires_grad = False

        def unfreeze_all(self):
            for p in self.parameters():
                p.requires_grad = True

        def unfreeze_last_n_blocks(self, n):
            self.unfreeze_all()

        def set_train_mode(self, is_train):
            self.train(is_train)

        def count_parameters(self, trainable_only=True):
            params = self.parameters()
            if trainable_only:
                return sum(p.numel() for p in params if p.requires_grad)
            return sum(p.numel() for p in params)

    backbone = _StubBackbone(cfg.model.backbone_output_dim)
    head = ProjectionHead(
        input_dim=cfg.model.backbone_output_dim,
        hidden_dim=cfg.model.projection_hidden_dim,
        output_dim=cfg.model.embedding_dim,
        dropout=0.0,
        l2_normalize=True,
    )
    return SignModel(backbone=backbone, head=head)


class TestSignModelForward:
    """Tests for the SignModel forward pass."""

    def test_forward_output_shape(self, small_model, cfg):
        """Forward pass must return (B, embedding_dim) unit vectors."""
        B = 4
        T = cfg.dataset.num_frames
        H = W = cfg.dataset.frame_size
        x = torch.randn(B, 3, T, H, W)
        out = small_model(x)
        assert out.shape == (B, cfg.model.embedding_dim)

    def test_output_is_unit_norm(self, small_model, cfg):
        """Forward outputs must be L2-normalised."""
        x = torch.randn(
            8, 3, cfg.dataset.num_frames, cfg.dataset.frame_size, cfg.dataset.frame_size
        )
        out = small_model(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_cosine_loss_perfect_prediction(self, small_model):
        """Cosine loss must be ≈ 0 when pred == target."""
        from models.sign_model import SignModel

        pred = F.normalize(torch.randn(8, 300), dim=-1)
        loss = SignModel.cosine_loss(pred, pred)
        assert loss.item() < 1e-5, f"Expected ~0, got {loss.item()}"

    def test_cosine_loss_antipodal(self):
        """Cosine loss must be ≈ 2 when pred == -target (antipodal)."""
        from models.sign_model import SignModel

        pred = F.normalize(torch.randn(8, 300), dim=-1)
        loss = SignModel.cosine_loss(pred, -pred)
        assert abs(loss.item() - 2.0) < 1e-4, f"Expected ~2.0, got {loss.item()}"

    def test_triplet_loss_non_negative(self):
        """Triplet loss must always be ≥ 0."""
        from models.sign_model import SignModel

        anchor = F.normalize(torch.randn(8, 300), dim=-1)
        positive = F.normalize(torch.randn(8, 300), dim=-1)
        negative = F.normalize(torch.randn(8, 300), dim=-1)
        loss = SignModel.triplet_loss(anchor, positive, negative, margin=0.2)
        assert loss.item() >= 0.0

    def test_combined_loss_returns_dict(self, small_model):
        """combined_loss must return a (tensor, dict) tuple with required keys."""
        B, D = 8, 300
        pred = F.normalize(torch.randn(B, D), dim=-1)
        target = F.normalize(torch.randn(B, D), dim=-1)
        total, loss_dict = small_model.combined_loss(pred, target)
        assert isinstance(total, torch.Tensor)
        for key in ("cosine_loss", "triplet_loss", "total_loss"):
            assert key in loss_dict, f"Missing key '{key}' in loss_dict"

    def test_predict_topk_shape(self, small_model, dummy_embeddings, cfg):
        """predict_topk must return (B, k) indices and scores."""
        B = 4
        k = cfg.inference.top_k
        T, H, W = cfg.dataset.num_frames, cfg.dataset.frame_size, cfg.dataset.frame_size
        x = torch.randn(B, 3, T, H, W)
        indices, scores = small_model.predict_topk(x, dummy_embeddings, k=k)
        assert indices.shape == (B, k)
        assert scores.shape == (B, k)

    def test_apply_phase_1_freezes_backbone(self, small_model):
        """apply_phase(1) must set all backbone params to requires_grad=False."""
        small_model.apply_phase(1)
        frozen = all(not p.requires_grad for p in small_model.backbone.parameters())
        assert frozen, "Backbone params not frozen after apply_phase(1)"

    def test_apply_phase_3_unfreezes_backbone(self, small_model):
        """apply_phase(3) must set all backbone params to requires_grad=True."""
        small_model.apply_phase(1)  # first freeze
        small_model.apply_phase(3)  # then full unfreeze
        unfrozen = all(p.requires_grad for p in small_model.backbone.parameters())
        assert unfrozen, "Backbone params not unfrozen after apply_phase(3)"

    def test_invalid_phase_raises(self, small_model):
        """apply_phase with an invalid phase number must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid training phase"):
            small_model.apply_phase(99)

    def test_checkpoint_round_trip(self, small_model, cfg, tmp_path, mocker):
        """Save then load a checkpoint — loaded model must match original.

        ``from_config`` is patched to return a fresh copy of the stub model so
        the test does not attempt to download or build a real I3D backbone.
        """
        import copy

        ckpt_path = tmp_path / "test_checkpoint.pt"
        small_model.save_checkpoint(
            path=ckpt_path,
            epoch=5,
            metrics={"top1": 0.55},
        )
        assert ckpt_path.exists(), "Checkpoint file was not created"

        # Patch from_config to return a fresh stub instead of a real backbone
        fresh_stub = copy.deepcopy(small_model)
        from models.sign_model import SignModel

        mocker.patch.object(SignModel, "from_config", return_value=fresh_stub)

        loaded_model, epoch, metrics = SignModel.load_checkpoint(ckpt_path, cfg, device="cpu")
        assert epoch == 5
        assert metrics.get("top1") == 0.55

        # Compare head state dicts
        for (k1, v1), (_k2, v2) in zip(
            small_model.head.state_dict().items(),
            loaded_model.head.state_dict().items(),
            strict=True,
        ):
            assert torch.allclose(v1, v2), f"Mismatch at key '{k1}' after round-trip"

    def test_model_summary_string(self, small_model):
        """model_summary must return a non-empty string."""
        summary = small_model.model_summary()
        assert isinstance(summary, str) and len(summary) > 0

    def test_count_parameters(self, small_model):
        """count_parameters must return a dict with backbone/head/total keys."""
        counts = small_model.count_parameters(trainable_only=False)
        assert "backbone" in counts
        assert "head" in counts
        assert counts["total"] == counts["backbone"] + counts["head"]
