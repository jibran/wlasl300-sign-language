"""Sign language model with a temporal transformer neck.

Backbone → Temporal Neck → Classifier Head, where the neck replaces the
naive global average pool with a transformer encoder that contextualises
every frame against every other frame before the clip-level vector is formed.

Architecture::

    (B, 3, 16, 256, 256)
        ↓  I3DBackbone.forward_features()   (feature_blocks only, no pool)
    (B, 2048, T', H', W')                  T'=2, H'=W'=8 for i3d_r50 / 16 frames
        ↓  TemporalNeck
           spatial avg pool  →  (B, T', 2048)
           input_proj        →  (B, T', 256)
           prepend [CLS]     →  (B, T'+1, 256)
           + sinusoidal PE
           TransformerEncoder × 2
           CLS output + LayerNorm  →  (B, 256)
        ↓  ClassifierHead  (best-performing head from phase-1 experiments)
           Linear 256→256 · BN · ReLU · Dropout(0.4) · Linear 256→300
    (B, 300)  ← raw logits

The three-phase training schedule is extended for three components:

- **Phase 1** — backbone frozen; neck + head train from scratch at lr=1e-3
- **Phase 2** — last 2 backbone blocks + neck + head at lr=1e-4 / 1e-5
- **Phase 3** — full backbone + neck + head at lr=1e-5 / 1e-6

Usage::

    uv run python train/train_classifier.py \\
        --config config/config.yaml --model temporal

    uv run python inference/inference_classifier.py \\
        --checkpoint trained_models/temporal/best/checkpoint.pt \\
        --model temporal --video WLASL300/0/00412.mp4
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from models.classifier_head import ClassifierHead
from models.i3d_backbone import I3DBackbone
from models.temporal_neck import TemporalNeck

log = logging.getLogger(__name__)


class SignModelTemporal(torch.nn.Module):
    """I3D backbone + Temporal Transformer neck + FC classifier head.

    Args:
        backbone: Pretrained :class:`~models.i3d_backbone.I3DBackbone`.
        neck: :class:`~models.temporal_neck.TemporalNeck` that converts the
            backbone's 5-D feature map to a single clip-level vector.
        head: :class:`~models.classifier_head.ClassifierHead` mapping the
            neck output to class logits.

    Example::

        cfg = Config.from_yaml("config/config.yaml")
        model = SignModelTemporal.from_config(cfg)
        model.apply_phase(1)

        x = torch.randn(2, 3, 16, 256, 256)
        logits = model(x)               # (2, 300)
        loss = model.loss(logits, labels)
    """

    def __init__(
        self,
        backbone: I3DBackbone,
        neck: TemporalNeck,
        head: ClassifierHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: Tensor) -> Tensor:
        """Run backbone → neck → head.

        Args:
            x: Video clips ``(B, 3, T, H, W)``, float32, ImageNet-normalised.

        Returns:
            Raw logit tensor ``(B, num_classes)``, float32.
        """
        features = self.backbone.forward_features(x)  # (B, C, T', H', W')
        clip_vec = self.neck(features)  # (B, d_model)
        logits = self.head(clip_vec)  # (B, num_classes)
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
        """Cross-entropy loss with optional label smoothing.

        Args:
            logits: ``(B, num_classes)`` raw logits.
            labels: ``(B,)`` ground-truth class indices, int64.
            label_smoothing: Smoothing factor in ``[0, 1)``.

        Returns:
            Scalar loss tensor.
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
                zip(top_indices[b].tolist(), top_probs[b].tolist(), strict=True),
                start=1,
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
    # Training phase management
    # ---------------------------------------------------------------------- #

    def apply_phase(self, phase: int, cfg: object | None = None) -> None:
        """Switch freeze/unfreeze state for a training phase.

        The neck and head are always trainable.  Only the backbone freeze
        state changes across phases:

        - **Phase 1** — backbone frozen; neck + head train from scratch.
        - **Phase 2** — last N backbone blocks unfrozen.
        - **Phase 3** — full backbone unfrozen.

        Args:
            phase: Training phase — 1, 2, or 3.
            cfg: Optional config for reading ``unfreeze_last_n_blocks``.

        Raises:
            ValueError: If ``phase`` is not 1, 2, or 3.
        """
        if phase == 1:
            self.backbone.freeze()
            log.info("SignModelTemporal: Phase 1 — backbone frozen, " "neck + head training")
        elif phase == 2:
            n_blocks = 2
            if cfg is not None:
                n_blocks = cfg.training.phase2.unfreeze_last_n_blocks or 2
            self.backbone.unfreeze_last_n_blocks(n_blocks)
            log.info(
                "SignModelTemporal: Phase 2 — last %d backbone blocks "
                "unfrozen, neck + head training",
                n_blocks,
            )
        elif phase == 3:
            self.backbone.unfreeze_all()
            log.info("SignModelTemporal: Phase 3 — full backbone unfrozen")
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
        """Save model state and metadata to a ``.pt`` file.

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
            "model_type": "temporal",
            "model_state_dict": self.state_dict(),
            "optimiser_state_dict": optimiser_state,
            "scheduler_state_dict": scheduler_state,
            "metrics": metrics or {},
            "config": cfg.to_dict() if cfg is not None else {},
            "backbone_name": self.backbone.model_name,
            "neck_output_dim": self.neck.output_dim,
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
    ) -> tuple[SignModelTemporal, int, dict]:
        """Load a :class:`SignModelTemporal` from a saved checkpoint.

        Args:
            path: Path to the ``.pt`` checkpoint file.
            cfg: Config describing the model architecture.
            device: Target device string (e.g. ``"cuda"``, ``"cpu"``).
            strict: Require exact state dict match.

        Returns:
            Tuple ``(model, epoch, metrics)``.

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
        """Count parameters per component.

        Args:
            trainable_only: Count only trainable parameters.

        Returns:
            Dict with ``"backbone"``, ``"neck"``, ``"head"``, ``"total"`` keys.
        """
        b = self.backbone.count_parameters(trainable_only)
        n = self.neck.count_parameters(trainable_only)
        h = self.head.count_parameters(trainable_only)
        return {"backbone": b, "neck": n, "head": h, "total": b + n + h}

    def model_summary(self) -> str:
        """Return a concise human-readable model summary.

        Returns:
            Multi-line string with architecture and parameter counts.
        """
        counts = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        lines = [
            "=" * 60,
            "  SignModelTemporal summary",
            "=" * 60,
            f"  Backbone        : {self.backbone.model_name}",
            f"  Backbone output : {self.backbone.output_dim}  (pre-pool)",
            f"  Neck            : TemporalTransformer  →  {self.neck.output_dim}",
            f"  Head            : {self.neck.output_dim}"
            f" → {self.head.hidden_dim} → {self.head.num_classes}",
            f"  Neck params     : {counts['neck']:,}",
            f"  Head params     : {counts['head']:,}",
            f"  Total params    : {counts['total']:,}",
            f"  Trainable params: {trainable['total']:,}",
            f"    ↳ backbone    : {trainable['backbone']:,}",
            f"    ↳ neck        : {trainable['neck']:,}",
            f"    ↳ head        : {trainable['head']:,}",
            "=" * 60,
        ]
        return "\n".join(lines)

    # ---------------------------------------------------------------------- #
    # Factory constructor
    # ---------------------------------------------------------------------- #

    @classmethod
    def from_config(cls, cfg: object) -> SignModelTemporal:
        """Construct a :class:`SignModelTemporal` from a Config object.

        Reads architecture settings from ``cfg.model``,
        ``cfg.dataset.num_classes``, and ``cfg.temporal_neck`` (if present).

        Args:
            cfg: :class:`~config.base_config.Config` instance.

        Returns:
            Freshly initialised :class:`SignModelTemporal`.
        """
        model_cfg = cfg.model

        backbone = I3DBackbone(
            model_name=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            output_dim=model_cfg.backbone_output_dim,
        )

        # Read temporal neck config if present, otherwise use defaults
        tneck_cfg = getattr(cfg, "temporal_neck", None)
        neck = TemporalNeck(
            backbone_dim=backbone.output_dim,
            d_model=getattr(tneck_cfg, "d_model", 256),
            nhead=getattr(tneck_cfg, "nhead", 8),
            num_layers=getattr(tneck_cfg, "num_layers", 2),
            dim_feedforward=getattr(tneck_cfg, "dim_feedforward", 1024),
            dropout=getattr(tneck_cfg, "dropout", 0.1),
        )

        # hidden_dim matches neck.output_dim (d_model) so the head is
        # d_model → d_model → num_classes rather than d_model → 512 → num_classes.
        # Using projection_hidden_dim (512) here would make the head wider than
        # its input, which adds parameters without benefit and breaks the
        # intended compression flow: 2048 → d_model → d_model → 300.
        head = ClassifierHead(
            input_dim=neck.output_dim,
            hidden_dim=neck.output_dim,
            num_classes=cfg.dataset.num_classes,
            dropout=model_cfg.dropout,
        )

        model = cls(backbone=backbone, neck=neck, head=head)
        log.info("\n%s", model.model_summary())
        return model
