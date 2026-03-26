"""Three-layer deep classification head for the i3d_r50 backbone.

Designed specifically for the pytorchvideo ``i3d_r50`` backbone whose global
average pool output is 2048-dimensional.  The head progressively reduces the
feature dimension before the final classification layer:

    Linear(2048 → 1024, bias=False) → BatchNorm1d → ReLU → Dropout(0.4)
    Linear(1024 → 1024, bias=False) → BatchNorm1d → ReLU → Dropout(0.4)
    Linear(1024 → 512,  bias=False) → BatchNorm1d → ReLU → Dropout(0.3)
    Linear(512  → 512,  bias=False) → BatchNorm1d → ReLU → Dropout(0.3)
    Linear(512  → 300,  bias=True)

All intermediate ``Linear`` layers use ``bias=False`` because ``BatchNorm1d``
follows each one and provides its own learnable shift (β), making a separate
bias redundant.  Only the final classification layer keeps ``bias=True``.

Parameter count (2048 → 2048 → 1024 → 1024 → 512 → 300):

- fc1: 2048 × 1024 = 2,097,152
- bn1: 1024 × 2    = 2,048
- fc2: 1024 × 512  = 524,288
- bn2: 512 × 2    = 1,024
- fc3: 512 × 512  = 262,144
- bn3: 512 × 2    = 1,024
- fc4: 512 × 512  = 262,144
- bn4: 512 × 2    = 1,024
- fc5: 512 × 300 + 300 = 153,900
- **Total: 3,208,748 params**

Example::

    from models.deep_classifier_head import DeepClassifierHead

    head = DeepClassifierHead(input_dim=2048, num_classes=300)
    logits = head(torch.randn(8, 2048))   # (8, 300)
    loss = F.cross_entropy(logits, labels)
"""

from __future__ import annotations

import logging

import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


class DeepClassifierHead(nn.Module):
    """Three-layer fully-connected classification head.

    Architecture::

        Linear(input_dim → hidden1, bias=False)
        BatchNorm1d(hidden1)
        ReLU
        Dropout(p=dropout1)
        Linear(hidden1 → hidden2, bias=False)
        BatchNorm1d(hidden2)
        ReLU
        Dropout(p=dropout2)
        Linear(hidden2 → num_classes, bias=True)

    Args:
        input_dim: Backbone output dimension.  Defaults to ``2048`` for
            pytorchvideo ``i3d_r50``.
        hidden1: Width of the first hidden layer.  Defaults to ``1024``.
        hidden2: Width of the second hidden layer.  Defaults to ``512``.
        num_classes: Number of output classes.  Defaults to ``300`` for
            WLASL300.
        dropout1: Dropout probability after the first hidden layer.
            Defaults to ``0.4``.
        dropout2: Dropout probability after the second hidden layer.
            Defaults to ``0.3``.

    Example::

        head = DeepClassifierHead(input_dim=2048, num_classes=300)
        logits = head(torch.randn(8, 2048))  # (8, 300)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden1: int = 1024,
        hidden2: int = 512,
        num_classes: int = 300,
        dropout1: float = 0.4,
        dropout2: float = 0.3,
    ) -> None:
        """Build the deep classification head.

        Args:
            input_dim: Backbone output dimension.
            hidden1: First hidden layer width.
            hidden2: Second hidden layer width.
            num_classes: Number of output classes.
            dropout1: Dropout probability after the first hidden layer.
            dropout2: Dropout probability after the second hidden layer.
        """
        super().__init__()

        self._input_dim = input_dim
        self._hidden1 = hidden1
        self._hidden2 = hidden2
        self._num_classes = num_classes
        self._dropout1 = dropout1
        self._dropout2 = dropout2

        # Block 1: 2048 → 1024
        self.fc1 = nn.Linear(input_dim, hidden1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=dropout1)

        # Block 2: 1024 → 1024
        self.fc2 = nn.Linear(hidden1, hidden1, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden1)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(p=dropout1)

        # Block 3: 1024 → 512
        self.fc3 = nn.Linear(hidden1, hidden2, bias=False)
        self.bn3 = nn.BatchNorm1d(hidden2)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(p=dropout2)

        # Block 4: 512 → 512
        self.fc4 = nn.Linear(hidden2, hidden2, bias=False)
        self.bn4 = nn.BatchNorm1d(hidden2)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop4 = nn.Dropout(p=dropout2)

        # Classifier: 512 → num_classes
        self.fc5 = nn.Linear(hidden2, num_classes, bias=True)

        self._init_weights()

        log.info(
            "DeepClassifierHead: %d → %d (drop=%.1f) → %d (drop=%.1f) → %d",
            input_dim,
            hidden1,
            dropout1,
            hidden2,
            dropout2,
            num_classes,
        )

    # ---------------------------------------------------------------------- #
    # Weight initialisation
    # ---------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        """Initialise all fc layers with Kaiming normal; zero fc3 bias."""
        for fc in (self.fc1, self.fc2, self.fc3, self.fc4, self.fc5):
            nn.init.kaiming_normal_(fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc5.bias)

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: Tensor) -> Tensor:
        """Run the deep classification head forward pass.

        Args:
            x: Backbone feature tensor ``(B, input_dim)``, float32.

        Returns:
            Raw logit tensor ``(B, num_classes)``, float32.  Apply
            ``F.cross_entropy`` directly — no softmax is applied here.
        """
        # Block 1
        x = self.fc1(x)  # (B, 1024)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        # Block 2
        x = self.fc2(x)  # (B, 1024)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        # Block 3
        x = self.fc3(x)  # (B, 512)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        # Block 4
        x = self.fc4(x)  # (B, 512)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.drop4(x)

        # Classifier
        x = self.fc5(x)  # (B, num_classes)
        return x

    # ---------------------------------------------------------------------- #
    # Introspection
    # ---------------------------------------------------------------------- #

    @property
    def input_dim(self) -> int:
        """Backbone feature dimension expected by this head."""
        return self._input_dim

    @property
    def hidden1(self) -> int:
        """First hidden layer width."""
        return self._hidden1

    @property
    def hidden2(self) -> int:
        """Second hidden layer width."""
        return self._hidden2

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
