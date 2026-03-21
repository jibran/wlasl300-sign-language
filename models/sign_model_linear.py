"""Sign language linear probe model composing I3D backbone and a single FC layer.

Provides :class:`SignModelLinear` — the simplest possible classification
model: the I3D backbone followed by a single ``Linear(backbone_dim → 300)``
layer.  No hidden layers, no batch norm, no dropout.

This model is primarily useful as a **linear probe baseline** — if the
pretrained I3D features are good, even a single linear layer should achieve
reasonable accuracy.  Comparing it against
:class:`~models.sign_model_classifier.SignModelClassifier` (deep FC head) and
:class:`~models.sign_model.SignModel` (Word2Vec embedding) quantifies how much
each architectural component contributes.

Architecture::

    (B, 3, 16, 256, 256)
        ↓  I3DBackbone  (i3d_r50, Kinetics-400 pretrained)
    (B, 1024)
        ↓  LinearHead
           Linear 1024 → 300
    (B, 300)  ← raw logits

Training uses cross-entropy loss.  Inference uses direct argmax / top-k.

Usage::

    uv run python train/train_linear.py --config config/config.yaml

    uv run python inference/inference_classifier.py \\
        --checkpoint trained_models/linear/best/checkpoint.pt \\
        --video WLASL300/0/00412.mp4 --top_k 5
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from models.i3d_backbone import I3DBackbone
from models.linear_head import LinearHead

log = logging.getLogger(__name__)


class SignModelLinear(torch.nn.Module):
    """I3D backbone + single linear classification layer.

    Args:
        backbone: Pretrained :class:`~models.i3d_backbone.I3DBackbone`.
        head: :class:`~models.linear_head.LinearHead` mapping backbone
            features directly to class logits.

    Example::

        cfg = Config.from_yaml("config/config.yaml")
        model = SignModelLinear.from_config(cfg)
        model.apply_phase(1)

        x = torch.randn(2, 3, 16, 256, 256)
        logits = model(x)               # (2, 300)
        loss = model.loss(logits, labels)
    """

    def __init__(self, backbone: I3DBackbone, head: LinearHead) -> None:
        """Compose backbone and linear head.

        Args:
            backbone: I3D feature extractor.
            head: Single linear classification layer.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: Tensor) -> Tensor:
        """Run the full forward pass from video clip to class logits.

        Args:
            x: Input tensor ``(B, 3, T, H, W)``, float32, ImageNet-normalised.

        Returns:
            Raw logit tensor ``(B, num_classes)``, float32.
        """
        features = self.backbone(x)  # (B, backbone_output_dim)
        logits = self.head(features)  # (B, num_classes)
        return logits

    # ---------------------------------------------------------------------- #
    # Loss
    # ---------------------------------------------------------------------- #

    @staticmethod
    def loss(
        logits: Tensor,
        labels: Tensor,
        label_smoothing: float = 0.1,
    ) -> Tensor:
        """Compute cross-entropy loss with optional label smoothing.

        Args:
            logits: Raw logit tensor ``(B, num_classes)``.
            labels: Ground-truth class indices ``(B,)``, int64.
            label_smoothing: Label smoothing factor in ``[0, 1)``.

        Returns:
            Scalar cross-entropy loss.
        """
        return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

    # ---------------------------------------------------------------------- #
    # Prediction
    # ---------------------------------------------------------------------- #

    def predict_topk(
        self,
        x: Tensor,
        vocab: list[str],
        k: int = 5,
    ) -> list[list[dict]]:
        """Predict top-k classes for a batch of video clips.

        Args:
            x: Input video clips ``(B, 3, T, H, W)``.
            vocab: Ordered list of class label strings.
            k: Number of top predictions per sample.

        Returns:
            List of length ``B``, each element a list of ``k`` dicts::

                [{"rank": 1, "label": "book", "score": 0.94}, ...]

            Scores are softmax probabilities in ``[0, 1]``.
        """
        logits = self(x)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(k=k, dim=-1)

        results: list[list[dict]] = []
        for b in range(logits.shape[0]):
            preds = []
            for rank, (idx, score) in enumerate(
                zip(top_indices[b].tolist(), top_probs[b].tolist(), strict=True), start=1
            ):
                preds.append(
                    {
                        "rank": rank,
                        "label": vocab[idx] if idx < len(vocab) else str(idx),
                        "score": round(float(score), 4),
                    }
                )
            results.append(preds)
        return results

    # ---------------------------------------------------------------------- #
    # Phase management
    # ---------------------------------------------------------------------- #

    def apply_phase(self, phase: int, cfg: object | None = None) -> None:
        """Switch the model's freeze/unfreeze state for a training phase.

        Same three-phase schedule as the other models:

        - **Phase 1**: Backbone frozen, linear head only.
        - **Phase 2**: Last ``N`` backbone blocks unfrozen.
        - **Phase 3**: Full backbone unfrozen.

        Args:
            phase: Training phase — 1, 2, or 3.
            cfg: Optional config for reading ``unfreeze_last_n_blocks``.

        Raises:
            ValueError: If ``phase`` is not 1, 2, or 3.
        """
        if phase == 1:
            self.backbone.freeze()
            log.info("SignModelLinear: Phase 1 — backbone frozen, linear head only")
        elif phase == 2:
            n_blocks = 2
            if cfg is not None:
                n_blocks = cfg.training.phase2.unfreeze_last_n_blocks or 2
            self.backbone.unfreeze_last_n_blocks(n_blocks)
            log.info(
                "SignModelLinear: Phase 2 — last %d backbone blocks unfrozen",
                n_blocks,
            )
        elif phase == 3:
            self.backbone.unfreeze_all()
            log.info("SignModelLinear: Phase 3 — full backbone unfrozen")
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
        """Serialise model state and metadata to a ``.pt`` file.

        Args:
            path: Destination path.
            epoch: Current epoch (0-indexed).
            optimiser_state: Optimiser state dict for resuming.
            scheduler_state: Scheduler state dict for resuming.
            metrics: Validation metrics dict.
            cfg: Config object for reproducibility.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_type": "linear",
            "model_state_dict": self.state_dict(),
            "optimiser_state_dict": optimiser_state,
            "scheduler_state_dict": scheduler_state,
            "metrics": metrics or {},
            "config": cfg.to_dict() if cfg is not None else {},
            "backbone_name": self.backbone.model_name,
            "num_classes": self.head.num_classes,
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
    ) -> tuple[SignModelLinear, int, dict]:
        """Load a :class:`SignModelLinear` from a saved checkpoint.

        Args:
            path: Path to the ``.pt`` checkpoint file.
            cfg: Config describing the model architecture.
            device: Target device.
            strict: Require exact state dict match.

        Returns:
            Tuple of ``(model, epoch, metrics)``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
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
        """Count parameters in backbone and head.

        Args:
            trainable_only: Count only trainable parameters.

        Returns:
            Dict with ``"backbone"``, ``"head"``, and ``"total"`` keys.
        """
        b = self.backbone.count_parameters(trainable_only)
        h = self.head.count_parameters(trainable_only)
        return {"backbone": b, "head": h, "total": b + h}

    def model_summary(self) -> str:
        """Return a concise human-readable model summary.

        Returns:
            Multi-line string with architecture and parameter counts.
        """
        counts = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        lines = [
            "=" * 55,
            "  SignModelLinear summary",
            "=" * 55,
            f"  Backbone        : {self.backbone.model_name}",
            f"  Backbone output : {self.backbone.output_dim}",
            f"  Num classes     : {self.head.num_classes}",
            f"  Head params     : {self.head.count_parameters(False):,}  " f"(1 linear layer)",
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
    def from_config(cls, cfg: object) -> SignModelLinear:
        """Construct a :class:`SignModelLinear` from a Config object.

        Args:
            cfg: :class:`~config.base_config.Config` instance.

        Returns:
            Freshly initialised :class:`SignModelLinear`.
        """
        model_cfg = cfg.model

        backbone = I3DBackbone(
            model_name=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            output_dim=model_cfg.backbone_output_dim,
        )

        head = LinearHead(
            input_dim=backbone.output_dim,
            num_classes=cfg.dataset.num_classes,
        )

        model = cls(backbone=backbone, head=head)
        log.info("\n%s", model.model_summary())
        return model
