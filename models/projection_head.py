"""Projection head that maps I3D features into the Word2Vec embedding space.

This module defines :class:`ProjectionHead`, a small trainable ``nn.Module``
that takes a ``(B, backbone_output_dim)`` feature vector from the I3D backbone
and projects it into the same ``embedding_dim``-dimensional space as the
Word2Vec target vectors.

Architecture::

    Linear(backbone_dim → hidden_dim)
    BatchNorm1d(hidden_dim)
    ReLU
    Dropout(p)
    Linear(hidden_dim → embedding_dim)
    L2 normalise  ← unit-sphere constraint for cosine similarity loss

The L2 normalisation at the output ensures that both the predicted embedding
and the Word2Vec target are unit vectors, making cosine similarity numerically
well-behaved and the loss bounded in ``[0, 2]``.

Example::

    from models.projection_head import ProjectionHead

    head = ProjectionHead(
        input_dim=2048,
        hidden_dim=512,
        output_dim=300,
        dropout=0.4,
        l2_normalize=True,
    )
    features = torch.randn(4, 2048)
    embedding = head(features)     # (4, 300), unit vectors
"""

from __future__ import annotations

import logging

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """Two-layer MLP that projects backbone features into the embedding space.

    The head is intentionally lightweight — its small parameter count is what
    makes Phase 1 (frozen backbone) training stable and fast even with the
    tiny WLASL300 dataset.

    Args:
        input_dim: Dimensionality of the input feature vector from the
            backbone (e.g. 2048 for ``i3d_r50``).
        hidden_dim: Width of the intermediate linear layer (e.g. 512).
        output_dim: Dimensionality of the output embedding vector.  Should
            equal the Word2Vec vector size (300 for Google News).
        dropout: Dropout probability applied between the two linear layers.
            Higher values provide stronger regularisation for small datasets.
        l2_normalize: If ``True``, L2-normalise the output to unit length.
            Required for cosine similarity loss.  Only set to ``False`` for
            ablation experiments.

    Example::

        head = ProjectionHead(2048, 512, 300, dropout=0.4, l2_normalize=True)
        x = torch.randn(8, 2048)
        out = head(x)               # (8, 300), unit vectors
        print(out.norm(dim=-1))     # tensor([1., 1., ..., 1.])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.4,
        l2_normalize: bool = True,
    ) -> None:
        """Build the two-layer projection head.

        Args:
            input_dim: Input feature dimension from the backbone.
            hidden_dim: Hidden layer width.
            output_dim: Output embedding dimension (= Word2Vec vector size).
            dropout: Dropout probability (applied between the two linears).
            l2_normalize: Apply L2 normalisation to the output.

        Raises:
            ValueError: If any dimension argument is less than 1, or if
                ``dropout`` is not in ``[0, 1)``.
        """
        super().__init__()

        if any(d < 1 for d in (input_dim, hidden_dim, output_dim)):
            raise ValueError(
                "All dimension arguments must be ≥ 1. "
                f"Got input_dim={input_dim}, hidden_dim={hidden_dim}, "
                f"output_dim={output_dim}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self._l2_normalize = l2_normalize
        self._output_dim = output_dim

        # ------------------------------------------------------------------ #
        # Layer definitions
        # ------------------------------------------------------------------ #

        # First linear: expand or compress to hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)

        # BatchNorm1d stabilises training when the backbone is frozen and
        # only this head receives gradient updates
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.relu = nn.ReLU(inplace=True)

        # Dropout before the final projection — primary regulariser
        self.dropout = nn.Dropout(p=dropout)

        # Second linear: project down to the embedding dimension
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)

        self._init_weights()

        log.info(
            "ProjectionHead: %d → %d → %d  dropout=%.2f  l2_normalize=%s",
            input_dim,
            hidden_dim,
            output_dim,
            dropout,
            l2_normalize,
        )

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: Tensor) -> Tensor:
        """Project backbone features to the normalised embedding space.

        Args:
            x: Input feature tensor of shape ``(B, input_dim)``.

        Returns:
            Embedding tensor of shape ``(B, output_dim)``.  If
            ``l2_normalize=True``, each row is a unit vector (L2 norm = 1).
        """
        x = self.fc1(x)  # (B, hidden_dim)
        x = self.bn1(x)  # stabilise activations
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, output_dim)

        if self._l2_normalize:
            x = F.normalize(x, p=2, dim=-1)  # unit-sphere projection

        return x

    # ---------------------------------------------------------------------- #
    # Introspection helpers
    # ---------------------------------------------------------------------- #

    @property
    def output_dim(self) -> int:
        """Output embedding dimensionality.

        Returns:
            Integer output dimension set at construction.
        """
        return self._output_dim

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of (trainable) projection head parameters.

        Args:
            trainable_only: Count only ``requires_grad=True`` parameters if
                ``True``, all parameters otherwise.

        Returns:
            Integer parameter count.
        """
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    # ---------------------------------------------------------------------- #
    # Weight initialisation
    # ---------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        """Initialise linear layer weights with Kaiming uniform initialisation.

        Uses Kaiming (He) uniform initialisation for both linear layers, which
        is the standard choice for layers followed by ReLU activations.
        BatchNorm weight is initialised to 1 and bias to 0 (PyTorch default,
        but stated explicitly for clarity).
        """
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
