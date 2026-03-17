"""Full sign language recognition model composing backbone and projection head.

This module provides :class:`SignModel`, the top-level ``nn.Module`` used
throughout training, evaluation, and inference.  It composes the
:class:`~models.i3d_backbone.I3DBackbone` feature extractor with the
:class:`~models.projection_head.ProjectionHead` to produce a unit-norm
embedding vector for each input video clip.

The model also exposes:

- :meth:`SignModel.cosine_loss` — primary training objective.
- :meth:`SignModel.triplet_loss` — auxiliary objective for inter-class separation.
- :meth:`SignModel.predict_topk` — nearest-neighbour inference against the
  class embedding matrix.
- :meth:`SignModel.apply_phase` — single call to switch freeze/unfreeze state
  between training phases.
- :meth:`SignModel.from_config` — factory constructor from a
  :class:`~config.base_config.Config` object.
- :meth:`SignModel.save_checkpoint` / :meth:`SignModel.load_checkpoint` —
  checkpoint serialisation with full metadata.

Example::

    from config import Config
    from models.sign_model import SignModel

    cfg = Config.from_yaml("config/config.yaml")
    model = SignModel.from_config(cfg)
    model.apply_phase(1)  # freeze backbone, train head only

    video = torch.randn(4, 3, 64, 224, 224)
    pred_emb = model(video)             # (4, 300)

    target_emb = torch.randn(4, 300)
    target_emb = F.normalize(target_emb, dim=-1)
    loss = model.cosine_loss(pred_emb, target_emb)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.i3d_backbone import I3DBackbone
from models.projection_head import ProjectionHead

log = logging.getLogger(__name__)


class SignModel(nn.Module):
    """End-to-end sign language recognition model.

    Composes an :class:`~models.i3d_backbone.I3DBackbone` with a
    :class:`~models.projection_head.ProjectionHead` to map raw video clips
    ``(B, 3, T, H, W)`` into unit-norm embedding vectors ``(B, embedding_dim)``
    aligned with the Word2Vec embedding space.

    Args:
        backbone: Pretrained :class:`~models.i3d_backbone.I3DBackbone`
            feature extractor.
        head: :class:`~models.projection_head.ProjectionHead` that maps
            backbone features to the embedding space.

    Example::

        cfg = Config.from_yaml("config/config.yaml")
        model = SignModel.from_config(cfg)
        model.apply_phase(1)           # Phase 1: backbone frozen

        x = torch.randn(2, 3, 64, 224, 224)
        emb = model(x)                 # (2, 300)
    """

    def __init__(
        self,
        backbone: I3DBackbone,
        head: ProjectionHead,
    ) -> None:
        """Compose backbone and projection head into a single model.

        Args:
            backbone: I3D feature extractor.
            head: Projection head mapping features to the embedding space.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: Tensor) -> Tensor:
        """Run the full forward pass from video clip to embedding vector.

        Args:
            x: Input tensor of shape ``(B, 3, T, H, W)``, float32,
               ImageNet-normalised.

        Returns:
            Unit-norm embedding tensor of shape ``(B, embedding_dim)``,
            float32.  Each row has L2 norm = 1 (enforced by the projection
            head's normalisation layer).
        """
        features = self.backbone(x)  # (B, backbone_output_dim)
        embedding = self.head(features)  # (B, embedding_dim)
        return embedding

    # ---------------------------------------------------------------------- #
    # Loss functions
    # ---------------------------------------------------------------------- #

    @staticmethod
    def cosine_loss(pred: Tensor, target: Tensor) -> Tensor:
        """Compute mean cosine embedding loss over a batch.

        Loss = ``1 - mean(cosine_similarity(pred, target))``.

        Both ``pred`` and ``target`` are assumed to be L2-normalised unit
        vectors.  The loss is 0 when all predictions are identical to their
        targets and 2 in the worst case (antipodal vectors).

        Args:
            pred: Predicted embedding tensor ``(B, D)``, L2-normalised.
            target: Target Word2Vec embedding tensor ``(B, D)``, L2-normalised.

        Returns:
            Scalar loss tensor (mean over the batch).
        """
        similarity = F.cosine_similarity(pred, target, dim=-1)  # (B,)
        return (1.0 - similarity).mean()

    @staticmethod
    def triplet_loss(
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
        margin: float = 0.2,
    ) -> Tensor:
        """Compute batch triplet loss with cosine distance.

        Encourages ``cosine_dist(anchor, positive) + margin < cosine_dist(anchor, negative)``.

        All inputs are assumed to be L2-normalised.  Cosine distance is
        defined as ``1 - cosine_similarity``.

        Args:
            anchor: Predicted embedding for the query sample ``(B, D)``.
            positive: Predicted embedding for a different video of the same
                class ``(B, D)``.
            negative: Predicted embedding for a video of a different class
                ``(B, D)``.  Should be a hard negative (the most similar
                incorrect class) for best training signal.
            margin: Minimum required distance gap between positive and
                negative pairs.  Typical value: 0.2.

        Returns:
            Scalar triplet loss (mean of ``max(0, d_pos - d_neg + margin)``
            over the batch).
        """
        dist_pos = 1.0 - F.cosine_similarity(anchor, positive, dim=-1)  # (B,)
        dist_neg = 1.0 - F.cosine_similarity(anchor, negative, dim=-1)  # (B,)
        loss = F.relu(dist_pos - dist_neg + margin)
        return loss.mean()

    def combined_loss(
        self,
        pred: Tensor,
        target: Tensor,
        triplet_weight: float = 0.3,
        triplet_margin: float = 0.2,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the combined cosine + triplet loss for a training batch.

        The triplet loss uses in-batch hard negative mining: for each anchor,
        the hardest negative is the sample in the batch with the highest
        cosine similarity to the anchor that belongs to a different class.

        If all samples in the batch share the same class (degenerate batch),
        only the cosine loss is returned.

        Args:
            pred: Predicted embedding batch ``(B, D)``, L2-normalised.
            target: Word2Vec target embedding batch ``(B, D)``, L2-normalised.
            triplet_weight: Scalar weight applied to the triplet loss term.
                Set to 0.0 to disable triplet loss entirely.
            triplet_margin: Margin for the triplet loss.

        Returns:
            A tuple of:
                - ``total_loss``: Combined scalar loss tensor.
                - ``loss_dict``: Dict with keys ``"cosine_loss"``,
                  ``"triplet_loss"``, and ``"total_loss"`` for logging.
        """
        cos_loss = self.cosine_loss(pred, target)
        loss_dict: dict[str, float] = {"cosine_loss": cos_loss.item()}

        if triplet_weight <= 0.0:
            loss_dict["triplet_loss"] = 0.0
            loss_dict["total_loss"] = cos_loss.item()
            return cos_loss, loss_dict

        # In-batch hard negative mining using target embeddings as class proxies
        trip_loss = self._inbatch_triplet_loss(pred, target, triplet_margin)
        total = cos_loss + triplet_weight * trip_loss

        loss_dict["triplet_loss"] = trip_loss.item()
        loss_dict["total_loss"] = total.item()
        return total, loss_dict

    @staticmethod
    def _inbatch_triplet_loss(
        pred: Tensor,
        target: Tensor,
        margin: float,
    ) -> Tensor:
        """Mine hard negatives from the current batch and compute triplet loss.

        For each sample ``i`` in the batch:
        - **Positive**: the target embedding ``target[i]`` (same class proxy).
        - **Negative**: the target embedding ``target[j]`` where ``j`` has
          the highest cosine similarity to ``pred[i]`` among all samples
          whose target embedding differs from ``target[i]``.

        If no valid negative exists (all samples belong to the same class),
        returns a zero tensor.

        Args:
            pred: Predicted embeddings ``(B, D)``, L2-normalised.
            target: Target (Word2Vec) embeddings ``(B, D)``, L2-normalised.
            margin: Triplet margin.

        Returns:
            Scalar triplet loss tensor.
        """
        B = pred.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Pairwise similarity matrix between all predicted embeddings
        # sim_matrix[i, j] = cosine similarity(pred[i], pred[j])
        sim_matrix = pred @ pred.T  # (B, B)

        # Pairwise similarity between target embeddings to identify same-class pairs
        # target_sim[i, j] ≈ 1.0 if i and j are the same class
        target_sim = target @ target.T  # (B, B)

        # Mask for different-class pairs: sim < 0.99 (not near-identical targets)
        diff_class_mask = target_sim < 0.99  # (B, B)
        # Exclude self-similarity
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=pred.device)
        valid_neg_mask = diff_class_mask & eye_mask  # (B, B)

        if not valid_neg_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # For each anchor, find the hardest negative (highest sim, different class)
        # Set invalid positions to -inf so they are never selected
        masked_sim = sim_matrix.masked_fill(~valid_neg_mask, float("-inf"))
        hard_neg_indices = masked_sim.argmax(dim=1)  # (B,)
        hard_negatives = target[hard_neg_indices]  # (B, D)

        return SignModel.triplet_loss(pred, target, hard_negatives, margin)

    # ---------------------------------------------------------------------- #
    # Inference
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def predict_topk(
        self,
        x: Tensor,
        class_embeddings: Tensor,
        k: int = 5,
    ) -> tuple[Tensor, Tensor]:
        """Predict the top-k most similar classes for a batch of video clips.

        Runs the forward pass, then performs nearest-neighbour retrieval by
        computing cosine similarity between each predicted embedding and the
        full class embedding matrix.

        Args:
            x: Input video clips ``(B, 3, T, H, W)``.
            class_embeddings: Precomputed class embedding matrix
                ``(num_classes, embedding_dim)``, L2-normalised.
                Typically ``dataset.class_embedding_matrix``.
            k: Number of top predictions to return per sample.

        Returns:
            A tuple of:
                - ``top_indices``: ``(B, k)`` int64 tensor of class indices,
                  ordered by decreasing similarity.
                - ``top_scores``: ``(B, k)`` float32 tensor of cosine
                  similarity scores in ``[-1, 1]``.
        """
        pred_emb = self(x)  # (B, D)

        # (B, D) × (D, num_classes) → (B, num_classes)
        similarities = pred_emb @ class_embeddings.T

        top_scores, top_indices = similarities.topk(k=k, dim=-1, largest=True)
        return top_indices, top_scores

    # ---------------------------------------------------------------------- #
    # Training phase management
    # ---------------------------------------------------------------------- #

    def apply_phase(self, phase: int, cfg: object | None = None) -> None:
        """Switch the model's freeze/unfreeze state for a training phase.

        Applies the correct backbone freeze policy as defined in the project
        training schedule:

        - **Phase 1**: Backbone fully frozen; only the projection head trains.
        - **Phase 2**: Last ``N`` backbone blocks unfrozen (``N`` from
          ``cfg.training.phase2.unfreeze_last_n_blocks``).
        - **Phase 3**: Entire backbone unfrozen.

        Args:
            phase: Integer training phase — 1, 2, or 3.
            cfg: Optional :class:`~config.base_config.Config` for reading
                ``unfreeze_last_n_blocks`` from phase 2 config.  If ``None``,
                defaults to 2 blocks for phase 2.

        Raises:
            ValueError: If ``phase`` is not 1, 2, or 3.
        """
        if phase == 1:
            self.backbone.freeze()
            log.info("SignModel: Phase 1 — backbone frozen, head only")

        elif phase == 2:
            n_blocks = 2
            if cfg is not None:
                n_blocks = cfg.training.phase2.unfreeze_last_n_blocks or 2
            self.backbone.unfreeze_last_n_blocks(n_blocks)
            log.info("SignModel: Phase 2 — last %d backbone blocks unfrozen", n_blocks)

        elif phase == 3:
            self.backbone.unfreeze_all()
            log.info("SignModel: Phase 3 — full backbone unfrozen")

        else:
            raise ValueError(f"Invalid training phase {phase}. Must be 1, 2, or 3.")

    # ---------------------------------------------------------------------- #
    # Checkpointing
    # ---------------------------------------------------------------------- #

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        optimiser_state: dict | None = None,
        scheduler_state: dict | None = None,
        metrics: dict | None = None,
        cfg: object | None = None,
    ) -> None:
        """Serialise model state and training metadata to a checkpoint file.

        Saves the full model state dict along with enough metadata to resume
        training or reproduce the inference environment exactly.

        Args:
            path: Destination file path (e.g.
                ``"trained_models/best/checkpoint.pt"``).
            epoch: Current training epoch (0-indexed).
            optimiser_state: ``optimizer.state_dict()`` for resuming training.
                ``None`` if saving for inference only.
            scheduler_state: ``scheduler.state_dict()`` for resuming training.
            metrics: Dict of validation metrics at checkpoint time
                (e.g. ``{"val_top1": 0.62, "val_top5": 0.88}``).
            cfg: Config dict (via ``cfg.to_dict()``) stored for reproducibility.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimiser_state_dict": optimiser_state,
            "scheduler_state_dict": scheduler_state,
            "metrics": metrics or {},
            "config": cfg.to_dict() if cfg is not None else {},
            "backbone_name": self.backbone.model_name,
            "embedding_dim": self.head.output_dim,
        }
        torch.save(payload, path)
        log.info("Checkpoint saved → %s  (epoch=%d)", path, epoch)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        cfg: object,
        device: str | None = None,
        strict: bool = True,
    ) -> tuple[SignModel, int, dict]:
        """Load a :class:`SignModel` from a saved checkpoint file.

        Constructs a fresh model from ``cfg``, then loads the saved state dict
        into it.

        Args:
            path: Path to the ``.pt`` checkpoint file.
            cfg: :class:`~config.base_config.Config` describing the model
                architecture to reconstruct.
            device: Device to map the checkpoint onto (e.g. ``"cuda:0"``,
                ``"cpu"``).  If ``None``, uses the current default device.
            strict: If ``True``, the saved state dict must exactly match the
                model's parameter names.  Set to ``False`` to load partial
                checkpoints.

        Returns:
            A tuple of:
                - ``model``: :class:`SignModel` with weights loaded.
                - ``epoch``: The epoch at which the checkpoint was saved.
                - ``metrics``: The metrics dict stored in the checkpoint.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            RuntimeError: If the state dict is incompatible with the model
                (only raised when ``strict=True``).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(path, map_location=map_location, weights_only=False)

        model = cls.from_config(cfg)
        model.load_state_dict(payload["model_state_dict"], strict=strict)
        model.to(map_location)

        epoch: int = payload.get("epoch", 0)
        metrics: dict = payload.get("metrics", {})

        log.info(
            "Checkpoint loaded ← %s  (epoch=%d  metrics=%s)",
            path,
            epoch,
            metrics,
        )
        return model, epoch, metrics

    # ---------------------------------------------------------------------- #
    # Introspection
    # ---------------------------------------------------------------------- #

    def count_parameters(self, trainable_only: bool = True) -> dict[str, int]:
        """Count parameters in the backbone and head separately.

        Args:
            trainable_only: Count only trainable parameters if ``True``.

        Returns:
            Dict with keys ``"backbone"``, ``"head"``, and ``"total"``.
        """
        b = self.backbone.count_parameters(trainable_only)
        h = self.head.count_parameters(trainable_only)
        return {"backbone": b, "head": h, "total": b + h}

    def model_summary(self) -> str:
        """Return a concise human-readable model summary string.

        Returns:
            Multi-line string describing the model architecture and parameter
            counts, suitable for logging at the start of training.
        """
        counts = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        lines = [
            "=" * 55,
            "  SignModel summary",
            "=" * 55,
            f"  Backbone        : {self.backbone.model_name}",
            f"  Backbone output : {self.backbone.output_dim}",
            f"  Head output     : {self.head.output_dim}",
            f"  Total params    : {counts['total']:,}",
            f"  Trainable params: {trainable['total']:,}",
            f"    ↳ backbone    : {trainable['backbone']:,}",
            f"    ↳ head        : {trainable['head']:,}",
            "=" * 55,
        ]
        return "\n".join(lines)

    # ---------------------------------------------------------------------- #
    # Factory constructor
    # ---------------------------------------------------------------------- #

    @classmethod
    def from_config(cls, cfg: object) -> SignModel:
        """Construct a :class:`SignModel` from a Config object.

        Reads architecture settings from ``cfg.model`` and builds the
        backbone and projection head accordingly.

        Args:
            cfg: Fully populated :class:`~config.base_config.Config` instance.

        Returns:
            A :class:`SignModel` with freshly initialised weights (or
            Kinetics-400 pretrained backbone weights if ``cfg.model.pretrained``
            is ``True``).

        Example::

            cfg = Config.from_yaml("config/config.yaml")
            model = SignModel.from_config(cfg)
            print(model.model_summary())
        """
        model_cfg = cfg.model

        backbone = I3DBackbone(
            model_name=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            output_dim=model_cfg.backbone_output_dim,
        )

        # Use backbone.output_dim (probed at load time) not model_cfg.backbone_output_dim
        # (from config) — they differ when the torchvision fallback is used or when
        # pytorchvideo returns a different dim than the config expects.
        head = ProjectionHead(
            input_dim=backbone.output_dim,
            hidden_dim=model_cfg.projection_hidden_dim,
            output_dim=model_cfg.embedding_dim,
            dropout=model_cfg.dropout,
            l2_normalize=model_cfg.l2_normalize,
        )

        model = cls(backbone=backbone, head=head)
        log.info("\n%s", model.model_summary())
        return model
