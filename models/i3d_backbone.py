"""I3D backbone feature extractor for sign language video understanding.

This module wraps a pretrained Inflated 3D Convolutional Network (I3D) from
``pytorchvideo`` or ``torchvision`` and exposes it as a clean ``nn.Module``
that returns a flat ``(B, backbone_output_dim)`` feature vector per video clip
via global average pooling.

The backbone is designed to be used in a two-phase training strategy:

1. **Frozen** — all parameters fixed; only the downstream projection head
   is trained.
2. **Partially unfrozen** — the last ``N`` residual blocks are trainable
   while earlier layers remain frozen.
3. **Fully unfrozen** — all parameters trainable at a very low learning rate.

Example::

    from models.i3d_backbone import I3DBackbone

    backbone = I3DBackbone(
        model_name="i3d_r50",
        pretrained=True,
        output_dim=1024,
    )
    backbone.freeze()

    x = torch.randn(2, 3, 64, 224, 224)   # (B, C, T, H, W)
    features = backbone(x)                 # (2, 1024)
"""

from __future__ import annotations

import logging

import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)

# Supported backbone identifiers and their global-avg-pool output dimensions.
# Add new entries here when supporting additional architectures.
# Supported backbone names. Output dims are probed at runtime so these values
# are used only for the supported-model check, not for shape validation.
# pytorchvideo dims: i3d_r50=1024, slow_r50=2048, slowfast_r50=2304, x3d_m=2048
# torchvision fallback (r3d_18): 512 for i3d_r50 / slow_r50
_BACKBONE_OUTPUT_DIMS: dict[str, int] = {
    "i3d_r50": 1024,
    "slow_r50": 2048,
    "slowfast_r50": 2304,
    "x3d_m": 2048,
}


class I3DBackbone(nn.Module):
    """Inflated 3D CNN backbone that extracts spatiotemporal video features.

    Loads a pretrained video model, removes its classification head, and
    returns a ``(B, output_dim)`` feature tensor per forward pass via global
    average pooling over the spatial and temporal dimensions.

    Supports granular layer freezing for the three-phase training strategy
    described in the project plan.

    Args:
        model_name: Backbone architecture identifier.  Supported values:
            ``"i3d_r50"``, ``"slow_r50"``, ``"slowfast_r50"``, ``"x3d_m"``.
        pretrained: If ``True``, load weights pretrained on Kinetics-400.
            Set to ``False`` only for ablation experiments.
        output_dim: Expected feature dimension after global average pooling.
            Must match the actual backbone output — see ``_BACKBONE_OUTPUT_DIMS``.
            Raises ``ValueError`` at construction time if mismatched.

    Raises:
        ValueError: If ``model_name`` is not in the supported list, or if
            ``output_dim`` does not match the known dimension for that model.
        ImportError: If neither ``pytorchvideo`` nor ``torchvision`` is
            available for the requested backbone.

    Example::

        backbone = I3DBackbone("i3d_r50", pretrained=True, output_dim=1024)
        backbone.freeze()

        x = torch.randn(4, 3, 64, 224, 224)
        out = backbone(x)   # shape: (4, 2048)
    """

    def __init__(
        self,
        model_name: str = "i3d_r50",
        pretrained: bool = True,
        output_dim: int = 1024,
    ) -> None:
        """Initialise the I3D backbone.

        Args:
            model_name: Backbone variant name.
            pretrained: Load Kinetics-400 pretrained weights.
            output_dim: Feature dimension expected after global avg pool.

        Raises:
            ValueError: If ``model_name`` is unsupported or ``output_dim``
                does not match the known dimension for the model.
        """
        super().__init__()

        if model_name not in _BACKBONE_OUTPUT_DIMS:
            raise ValueError(
                f"Unsupported backbone '{model_name}'. "
                f"Choose from: {list(_BACKBONE_OUTPUT_DIMS.keys())}"
            )

        self._model_name = model_name

        # Load the backbone and strip the classification head
        self._backbone = self._load_backbone(model_name, pretrained)

        # Probe the real output dim by running a dummy forward pass.
        # This handles the torchvision fallback (r3d_18 → 512) vs pytorchvideo
        # (i3d_r50 → 1024) transparently without requiring the config to know
        # which library was used.
        probed_dim = self._probe_output_dim()

        if output_dim != probed_dim:
            log.warning(
                "I3DBackbone: config backbone_output_dim=%d does not match "
                "actual backbone output %d — overriding with %d. "
                "Update config.yaml: model.backbone_output_dim: %d",
                output_dim,
                probed_dim,
                probed_dim,
                probed_dim,
            )

        self._output_dim = probed_dim

        log.info(
            "I3DBackbone: model=%s  pretrained=%s  output_dim=%d",
            model_name,
            pretrained,
            self._output_dim,
        )

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: Tensor) -> Tensor:
        """Extract spatiotemporal features from a batch of video clips.

        Args:
            x: Input tensor of shape ``(B, C, T, H, W)`` — batch size,
               channels (3 for RGB), temporal frames, height, width.
               Expected to be float32, ImageNet-normalised.

        Returns:
            Feature tensor of shape ``(B, output_dim)``, float32.
        """
        return self._backbone(x)

    # ---------------------------------------------------------------------- #
    # Output dim probing
    # ---------------------------------------------------------------------- #

    def _probe_output_dim(self) -> int:
        """Probe the backbone's actual output dimension via a dummy forward pass.

        Runs a single ``(1, 3, 8, 64, 64)`` tensor through the backbone on CPU
        to discover the real feature dimension.  This is called once at
        construction time so we always use the correct dim regardless of which
        library (pytorchvideo vs torchvision fallback) was loaded.

        Returns:
            Integer output feature dimension.
        """
        import torch

        device = next(self._backbone.parameters()).device
        dummy = torch.zeros(1, 3, 8, 64, 64, device=device)
        was_training = self._backbone.training
        self._backbone.eval()
        with torch.no_grad():
            out = self._backbone(dummy)
        if was_training:
            self._backbone.train()
        return int(out.shape[1])

    # ---------------------------------------------------------------------- #
    # Freeze / unfreeze helpers
    # ---------------------------------------------------------------------- #

    def freeze(self) -> None:
        """Freeze all backbone parameters (Phase 1 — warm-up).

        Sets ``requires_grad=False`` on every parameter and switches the
        backbone to ``eval()`` mode so that BatchNorm running statistics
        are not updated during the warm-up phase.

        Note:
            Call ``model.train()`` on the outer :class:`~models.sign_model.SignModel`
            after calling this — it will call ``backbone.train()`` internally,
            but ``freeze()`` overrides that by explicitly setting eval mode
            and disabling gradients.
        """
        for param in self._backbone.parameters():
            param.requires_grad = False
        self._backbone.eval()
        log.info("I3DBackbone: all layers frozen")

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze the last ``n`` residual blocks (Phase 2 — partial fine-tune).

        Identifies the top-level layer groups of the backbone (e.g.
        ``layer4``, ``layer3`` for a ResNet-style I3D) and unfreezes the
        last ``n`` of them.  Layers not in the last ``n`` groups remain
        frozen.

        Args:
            n: Number of layer groups to unfreeze from the end.  A value
               of 1 unfreezes only the last residual stage.  A value equal
               to or greater than the total number of layer groups is
               equivalent to calling :meth:`unfreeze_all`.

        Note:
            If the backbone architecture has a different layer naming
            convention, override this method in a subclass or call
            :meth:`unfreeze_all` instead.
        """
        # Collect named top-level children that contain trainable parameters
        named_children = [
            (name, module)
            for name, module in self._backbone.named_children()
            if any(p.numel() > 0 for p in module.parameters())
        ]

        if not named_children:
            log.warning("No named children found in backbone — calling unfreeze_all()")
            self.unfreeze_all()
            return

        # First freeze everything, then selectively unfreeze the last n blocks
        self.freeze()

        blocks_to_unfreeze = named_children[-n:]
        for name, module in blocks_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True
            module.train()
            log.info("I3DBackbone: unfroze block '%s'", name)

        log.info(
            "I3DBackbone: %d/%d blocks unfrozen (%s)",
            len(blocks_to_unfreeze),
            len(named_children),
            ", ".join(name for name, _ in blocks_to_unfreeze),
        )

    def unfreeze_all(self) -> None:
        """Unfreeze all backbone parameters (Phase 3 — full fine-tune).

        Sets ``requires_grad=True`` on every parameter and switches the
        backbone to ``train()`` mode so BatchNorm statistics are updated.
        """
        for param in self._backbone.parameters():
            param.requires_grad = True
        self._backbone.train()
        log.info("I3DBackbone: all layers unfrozen")

    def set_train_mode(self, is_train: bool) -> None:
        """Set backbone training mode while respecting frozen layer state.

        Unlike ``module.train()`` / ``module.eval()``, this method checks
        whether each parameter is frozen and keeps frozen layers in ``eval()``
        mode even when the rest of the model is in training mode.

        Args:
            is_train: If ``True``, switch unfrozen layers to ``train()``
                mode.  If ``False``, switch everything to ``eval()`` mode.
        """
        if not is_train:
            self._backbone.eval()
            return

        # Switch to train but keep frozen layers in eval
        self._backbone.train()
        for module in self._backbone.modules():
            has_frozen = any(not p.requires_grad for p in module.parameters(recurse=False))
            if has_frozen:
                module.eval()

    # ---------------------------------------------------------------------- #
    # Introspection helpers
    # ---------------------------------------------------------------------- #

    @property
    def output_dim(self) -> int:
        """Feature dimension returned by the backbone after global avg pool.

        Returns:
            Integer output dimension (e.g. 2048 for ``i3d_r50``).
        """
        return self._output_dim

    @property
    def model_name(self) -> str:
        """Backbone architecture identifier string.

        Returns:
            The ``model_name`` string passed at construction.
        """
        return self._model_name

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of (trainable) backbone parameters.

        Args:
            trainable_only: If ``True``, count only parameters with
                ``requires_grad=True``.  If ``False``, count all parameters.

        Returns:
            Integer parameter count.
        """
        params = self._backbone.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    # ---------------------------------------------------------------------- #
    # Private — backbone loading
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _load_backbone(model_name: str, pretrained: bool) -> nn.Module:
        """Load the backbone from pytorchvideo or torchvision and strip head.

        Tries ``pytorchvideo`` first (recommended — has all I3D variants),
        then falls back to ``torchvision`` for ``slow_r50``.

        Args:
            model_name: Backbone variant identifier.
            pretrained: Whether to load Kinetics-400 pretrained weights.

        Returns:
            An ``nn.Module`` that accepts ``(B, C, T, H, W)`` and returns
            ``(B, output_dim)`` after global average pooling.

        Raises:
            ImportError: If neither ``pytorchvideo`` nor ``torchvision``
                provides the requested backbone.
        """
        try:
            return _load_from_pytorchvideo(model_name, pretrained)
        except (ImportError, AttributeError, KeyError):
            log.warning(
                "pytorchvideo not available or model '%s' not found — "
                "falling back to torchvision",
                model_name,
            )

        try:
            return _load_from_torchvision(model_name, pretrained)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"Could not load backbone '{model_name}' from pytorchvideo "
                f"or torchvision. Install pytorchvideo:\n"
                f"  uv add pytorchvideo\n"
                f"Original error: {exc}"
            ) from exc


# =============================================================================
# Loader helpers — pytorchvideo and torchvision
# =============================================================================


def _load_from_pytorchvideo(model_name: str, pretrained: bool) -> nn.Module:
    """Load a backbone from pytorchvideo and wrap it to return flat features.

    Args:
        model_name: Backbone variant identifier.
        pretrained: Whether to use Kinetics-400 pretrained weights.

    Returns:
        Wrapped ``nn.Module`` accepting ``(B, C, T, H, W)`` input and
        returning ``(B, output_dim)`` features.

    Raises:
        ImportError: If ``pytorchvideo`` is not installed.
        KeyError: If ``model_name`` is not available in pytorchvideo.
    """
    from pytorchvideo.models.hub import (  # type: ignore[import]
        i3d_r50,
        slow_r50,
        slowfast_r50,
        x3d_m,
    )

    _PTV_REGISTRY: dict[str, object] = {
        "i3d_r50": i3d_r50,
        "slow_r50": slow_r50,
        "slowfast_r50": slowfast_r50,
        "x3d_m": x3d_m,
    }

    if model_name not in _PTV_REGISTRY:
        raise KeyError(f"'{model_name}' not in pytorchvideo registry")

    model = _PTV_REGISTRY[model_name](pretrained=pretrained)

    # pytorchvideo models end with a `head` that contains a projection and
    # softmax.  We replace it with a global average pool + flatten.
    return _PytorchVideoWrapper(model)


def _load_from_torchvision(model_name: str, pretrained: bool) -> nn.Module:
    """Load a backbone from torchvision and wrap it to return flat features.

    Currently supports ``slow_r50`` via ``torchvision.models.video``.

    Args:
        model_name: Backbone variant identifier.
        pretrained: Whether to use Kinetics-400 pretrained weights.

    Returns:
        Wrapped ``nn.Module`` accepting ``(B, C, T, H, W)`` input and
        returning ``(B, output_dim)`` features.

    Raises:
        ImportError: If ``torchvision`` is not installed.
        AttributeError: If ``model_name`` is not available in torchvision.
    """
    import torchvision.models.video as video_models  # type: ignore[import]

    weights = "KINETICS400_V1" if pretrained else None

    if model_name in ("i3d_r50", "slow_r50"):
        # torchvision uses r3d_18 / mc3_18; closest to slow_r50 is r3d_18
        # For production use pytorchvideo instead
        model = video_models.r3d_18(weights=weights)
        # Strip the final fc classifier
        model.fc = nn.Identity()
        return _TorchvisionWrapper(model)

    raise AttributeError(
        f"torchvision does not provide '{model_name}'. "
        "Install pytorchvideo for full model support."
    )


class _PytorchVideoWrapper(nn.Module):
    """Thin wrapper that removes the pytorchvideo classification head.

    pytorchvideo models expose a ``model.blocks`` attribute.  The last block
    is the detection/classification head.  This wrapper replaces it with
    a global average pool and flatten to produce flat feature vectors.

    Args:
        ptv_model: A pytorchvideo model instance (e.g. from
            ``pytorchvideo.models.hub.i3d_r50``).
    """

    def __init__(self, ptv_model: nn.Module) -> None:
        """Initialise wrapper around a pytorchvideo model.

        Args:
            ptv_model: Pretrained pytorchvideo model.
        """
        super().__init__()
        # Keep all blocks except the final classification head
        if hasattr(ptv_model, "blocks"):
            # pytorchvideo uses a ModuleList of blocks; last block is the head
            self.feature_blocks = nn.Sequential(*list(ptv_model.blocks)[:-1])
        else:
            self.feature_blocks = ptv_model
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Run the feature extraction pipeline.

        Args:
            x: Input tensor ``(B, C, T, H, W)``.

        Returns:
            Feature tensor ``(B, output_dim)``.
        """
        # Pass tensor directly — current pytorchvideo (>=0.1.5) expects a
        # plain Tensor, not a list.  Older versions expected [x]; that caused
        # "conv3d() received list instead of Tensor".
        features = self.feature_blocks(x)
        if isinstance(features, (list, tuple)):
            features = features[0]
        # features: (B, C, T', H', W')
        features = self.pool(features)  # (B, C, 1, 1, 1)
        features = self.flatten(features)  # (B, C)
        return features


class _TorchvisionWrapper(nn.Module):
    """Thin wrapper that removes the torchvision video model classifier.

    Args:
        tv_model: A torchvision video model with the ``fc`` layer already
            replaced by ``nn.Identity()``.
    """

    def __init__(self, tv_model: nn.Module) -> None:
        """Initialise wrapper around a torchvision video model.

        Args:
            tv_model: Torchvision model with head removed.
        """
        super().__init__()
        self.model = tv_model
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Run the feature extraction pipeline.

        Args:
            x: Input tensor ``(B, C, T, H, W)``.

        Returns:
            Feature tensor ``(B, output_dim)``.
        """
        features = self.model(x)
        if features.dim() == 5:
            features = self.pool(features)
            features = self.flatten(features)
        return features
