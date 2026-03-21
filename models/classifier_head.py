"""Fully-connected classification head for sign language recognition.

Replaces the embedding-projection head in :mod:`models.projection_head` with a
standard deep classifier:

    Linear(backbone_dim → hidden_dim) → BN → ReLU → Dropout → Linear(hidden_dim → num_classes)

The output is a raw logit vector of shape ``(B, num_classes)``.  No L2
normalisation is applied — the caller applies ``F.cross_entropy`` directly on
the logits.

Example::

    from models.classifier_head import ClassifierHead

    head = ClassifierHead(input_dim=1024, hidden_dim=512, num_classes=300, dropout=0.4)
    logits = head(features)   # (B, 300)
    loss = F.cross_entropy(logits, labels)
"""

from __future__ import annotations

import logging

import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


class ClassifierHead(nn.Module):
    """Deep fully-connected classification head.

    Architecture::

        Linear(input_dim → hidden_dim)
        BatchNorm1d(hidden_dim)
        ReLU
        Dropout(p=dropout)
        Linear(hidden_dim → num_classes)

    Args:
        input_dim: Feature dimension from the backbone (e.g. 1024 for i3d_r50).
        hidden_dim: Width of the intermediate hidden layer.
        num_classes: Number of output classes (300 for WLASL300).
        dropout: Dropout probability applied after ReLU.

    Example::

        head = ClassifierHead(input_dim=1024, hidden_dim=512, num_classes=300)
        logits = head(torch.randn(8, 1024))  # (8, 300)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.4,
    ) -> None:
        """Build the classification head.

        Args:
            input_dim: Backbone output dimension.
            hidden_dim: Hidden layer width.
            num_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes
        self._dropout = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=True)

        self._init_weights()

        log.info(
            "ClassifierHead: %d → %d → %d  dropout=%.2f",
            input_dim,
            hidden_dim,
            num_classes,
            dropout,
        )

    # ---------------------------------------------------------------------- #
    # Weight initialisation
    # ---------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        """Initialise fc layers with Kaiming normal and zero biases."""
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: Tensor) -> Tensor:
        """Run the classification head forward pass.

        Args:
            x: Backbone feature tensor ``(B, input_dim)``, float32.

        Returns:
            Raw logit tensor ``(B, num_classes)``, float32.  Apply
            ``F.cross_entropy`` directly — no softmax is applied here.
        """
        x = self.fc1(x)  # (B, hidden_dim)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)  # (B, num_classes)
        return x

    # ---------------------------------------------------------------------- #
    # Introspection
    # ---------------------------------------------------------------------- #

    @property
    def input_dim(self) -> int:
        """Input feature dimension."""
        return self._input_dim

    @property
    def hidden_dim(self) -> int:
        """Hidden layer width."""
        return self._hidden_dim

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return self._num_classes

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count head parameters.

        Args:
            trainable_only: Count only parameters with ``requires_grad=True``.

        Returns:
            Integer parameter count.
        """
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)
