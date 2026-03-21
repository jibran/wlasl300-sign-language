"""Single linear classification head for sign language recognition.

The simplest possible classification head: one weight matrix from backbone
output dimension to num_classes, with no hidden layers, no batch norm, no
dropout, and no non-linearity.

    Linear(backbone_dim → num_classes)

This head is intentionally minimal — it serves as a baseline to measure how
much of the model's performance comes from the I3D backbone features alone,
without any additional capacity in the head.

Example::

    from models.linear_head import LinearHead

    head = LinearHead(input_dim=1024, num_classes=300)
    logits = head(features)   # (B, 300)
    loss = F.cross_entropy(logits, labels)
"""

from __future__ import annotations

import logging

import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


class LinearHead(nn.Module):
    """Single linear classification head.

    Architecture::

        Linear(input_dim → num_classes)

    No hidden layer, batch norm, dropout, or non-linearity.  This is the
    minimum viable head for classification — useful as a baseline to measure
    the I3D backbone's raw representational quality.

    Args:
        input_dim: Feature dimension from the backbone (e.g. 1024 for i3d_r50).
        num_classes: Number of output classes (300 for WLASL300).

    Example::

        head = LinearHead(input_dim=1024, num_classes=300)
        logits = head(torch.randn(8, 1024))  # (8, 300)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
    ) -> None:
        """Build the linear head.

        Args:
            input_dim: Backbone output dimension.
            num_classes: Number of output classes.
        """
        super().__init__()

        self._input_dim = input_dim
        self._num_classes = num_classes

        self.fc = nn.Linear(input_dim, num_classes, bias=True)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc.bias)

        log.info("LinearHead: %d → %d", input_dim, num_classes)

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: Tensor) -> Tensor:
        """Run the linear head forward pass.

        Args:
            x: Backbone feature tensor ``(B, input_dim)``, float32.

        Returns:
            Raw logit tensor ``(B, num_classes)``, float32.
        """
        return self.fc(x)

    # ---------------------------------------------------------------------- #
    # Introspection
    # ---------------------------------------------------------------------- #

    @property
    def input_dim(self) -> int:
        """Input feature dimension."""
        return self._input_dim

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
