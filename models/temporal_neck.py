"""Temporal Transformer neck for sign language video understanding.

Sits between the I3D backbone's per-frame spatial features and the
classification head, replacing the naive global average pool with a
transformer encoder that lets every frame attend to every other frame.

Architecture position::

    Backbone  →  (B, C, T', H', W')   [pre-pool 5-D feature map]
        ↓  spatial pool to frame tokens
    (B, T', C)
        ↓  TemporalNeck  (projection + [CLS] + positional encoding + transformer)
    (B, d_model)                       [CLS token output]
        ↓  ClassifierHead
    (B, num_classes)

Why a transformer over global average pool
------------------------------------------
Global average pool discards temporal order entirely — the pooled vector for
"book" looks the same whether the hand moves left-then-right or
right-then-left.  The transformer encoder attends across all T' frame tokens
simultaneously, so the final [CLS] vector encodes *how* the spatial features
evolve over time, not just *what* they are on average.

For 16 input frames, ``i3d_r50`` produces T'=2 temporal positions after its
3D convolutions.  The transformer therefore attends across 2 frame tokens
(+ 1 CLS token = 3 positions total).  When longer clips are used or a
backbone with finer temporal resolution is swapped in, the attention
automatically scales to the larger T'.

Components
----------
- **Input projection**: ``Linear(backbone_dim → d_model)`` — projects each
  frame token from backbone feature space into the transformer's working
  dimension.
- **[CLS] token**: a learnable vector prepended to the token sequence, whose
  output after the transformer is used as the clip-level representation
  (same as BERT / ViT).
- **Positional encoding**: sinusoidal, added to each token before the
  transformer.  Sinusoidal is used (rather than learnable) so the model
  generalises to different temporal lengths without retraining the
  positional embeddings.
- **Transformer encoder**: ``num_layers`` × standard multi-head self-attention
  blocks (``nn.TransformerEncoderLayer`` with ``batch_first=True``).
- **Layer norm**: applied to the [CLS] output before the head.

Parameter count (defaults: d_model=256, nhead=8, num_layers=2,
                 dim_feedforward=1024, backbone_dim=2048)::

    input_proj:   2048 × 256 + 256         =  524,544
    cls_token:    256                        =      256
    pos_encoding: (no parameters — sinusoidal)
    transformer:  2 × ~786K                 = 1,573,376  (approx)
    norm:         256 × 2                   =      512
    ─────────────────────────────────────────────────────
    Total neck:   ≈ 2,098,688 params

Example::

    from models.temporal_neck import TemporalNeck

    neck = TemporalNeck(backbone_dim=2048, d_model=256)
    # x: per-frame spatial features from backbone, shape (B, 2048, 2, 8, 8)
    out = neck(x)   # (B, 256)
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


class TemporalNeck(nn.Module):
    """Temporal Transformer neck.

    Accepts the raw 5-D feature map from the backbone (before global average
    pool), spatially pools each time step into a single token, then applies a
    transformer encoder with a [CLS] token to produce a single clip-level
    vector.

    Args:
        backbone_dim: Channel dimension of the backbone's pre-pool feature map
            (2048 for ``i3d_r50`` with pytorchvideo).
        d_model: Internal transformer dimension.  All attention and feedforward
            operations run at this width.  Defaults to ``256`` — large enough
            to capture temporal structure, small enough to fit in 6 GB VRAM
            when trained on a GTX 1660 Ti.
        nhead: Number of attention heads.  Must divide ``d_model`` evenly.
            Defaults to ``8``.
        num_layers: Number of stacked transformer encoder layers.  Defaults
            to ``2`` — sufficient for 16-frame clips where T'=2 after I3D.
            Increase to 4 on an A100.
        dim_feedforward: FFN hidden dimension inside each transformer layer.
            Defaults to ``1024`` (4× d_model).
        dropout: Dropout probability applied inside the transformer and on the
            positional encoding.  Defaults to ``0.1``.
        max_seq_len: Maximum temporal sequence length (T' + 1 for [CLS]).
            Sinusoidal positional encoding is pre-computed up to this length.
            Defaults to ``64`` — well above the T'=2 produced by i3d_r50 on
            16 frames, leaving headroom for longer clips.

    Example::

        neck = TemporalNeck(backbone_dim=2048, d_model=256, nhead=8, num_layers=2)
        x = torch.randn(4, 2048, 2, 8, 8)   # (B, C, T', H', W')
        out = neck(x)                         # (4, 256)
    """

    def __init__(
        self,
        backbone_dim: int = 2048,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()

        self._backbone_dim = backbone_dim
        self._d_model = d_model

        # Project each spatial-average frame token into transformer space
        self.input_proj = nn.Linear(backbone_dim, d_model)

        # Learnable [CLS] token — prepended to the frame token sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Sinusoidal positional encoding — not learnable, generalises to
        # unseen sequence lengths
        self.register_buffer(
            "pos_encoding",
            _build_sinusoidal_encoding(max_seq_len, d_model),
        )
        self.pos_dropout = nn.Dropout(p=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, T, d_model) — matches our layout
            norm_first=True,  # pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Final layer norm on CLS output before the head
        self.norm = nn.LayerNorm(d_model)

        log.info(
            "TemporalNeck: backbone_dim=%d  d_model=%d  nhead=%d  "
            "num_layers=%d  dim_feedforward=%d  dropout=%.2f",
            backbone_dim,
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            dropout,
        )

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: Tensor) -> Tensor:
        """Convert backbone feature map to a single clip-level vector.

        Args:
            x: 5-D feature map from backbone ``(B, C, T', H', W')``.
                This is the output of ``feature_blocks`` *before* global avg
                pool, as returned by
                :meth:`~models.i3d_backbone.I3DBackbone.forward_features`.

        Returns:
            Clip-level feature vector ``(B, d_model)``, float32.
        """
        B, C, T, H, W = x.shape

        # 1. Spatial average pool: collapse H × W into a single token per frame
        #    (B, C, T, H, W) → (B, C, T) → (B, T, C)
        tokens = x.mean(dim=(-2, -1))  # (B, C, T)
        tokens = tokens.permute(0, 2, 1)  # (B, T, C)

        # 2. Project into transformer dimension
        tokens = self.input_proj(tokens)  # (B, T, d_model)

        # 3. Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, T+1, d_model)

        # 4. Add sinusoidal positional encoding
        tokens = tokens + self.pos_encoding[:, : T + 1, :]
        tokens = self.pos_dropout(tokens)

        # 5. Transformer encoder
        tokens = self.transformer(tokens)  # (B, T+1, d_model)

        # 6. Extract [CLS] token output and apply layer norm
        cls_out = tokens[:, 0, :]  # (B, d_model)
        return self.norm(cls_out)

    # ---------------------------------------------------------------------- #
    # Introspection
    # ---------------------------------------------------------------------- #

    @property
    def output_dim(self) -> int:
        """Output feature dimension (= d_model)."""
        return self._d_model

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count neck parameters.

        Args:
            trainable_only: Count only parameters with ``requires_grad=True``.

        Returns:
            Integer parameter count.
        """
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)


# =============================================================================
# Positional encoding helper
# =============================================================================


def _build_sinusoidal_encoding(max_len: int, d_model: int) -> Tensor:
    """Build a sinusoidal positional encoding table.

    Uses the standard formula from *Attention is All You Need* (Vaswani 2017):

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Args:
        max_len: Maximum sequence length to pre-compute.
        d_model: Embedding / model dimension.

    Returns:
        Tensor of shape ``(1, max_len, d_model)`` — batch dimension prepended
        for broadcasting.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, max_len, d_model)
