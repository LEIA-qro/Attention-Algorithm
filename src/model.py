"""
model.py — DriverStateNet: Bi-LSTM with Temporal Attention
===========================================================

Architecture
------------
1. **LayerNorm** on the raw 18-D input.
2. **Bi-LSTM** (2 layers, hidden=64, dropout=0.3) processes the
   (batch, seq_len, 18) sequence.
3. **Temporal Attention** (additive / Bahdanau) collapses the LSTM
   output sequence into a single context vector.
4. **Classifier Head**: Linear(128→128) → ReLU → Dropout(0.4) →
   Linear(128→64) → ReLU → Dropout(0.4) → Linear(64→3).
5. ``forward()`` returns raw logits (B, 3).

The model is designed to stay near ~200 K parameters.

Public API
----------
- ``DriverStateNet(input_dim, hidden_dim, num_layers, num_classes, ...)``
- ``TemporalAttention(hidden_dim)``
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DriverStateNet", "TemporalAttention"]

logger = logging.getLogger(__name__)


# Temporal Attention (Bahdanau / Additive)

class TemporalAttention(nn.Module):
    """Additive (Bahdanau) attention over a temporal sequence.

    Given LSTM outputs ``H ∈ ℝ^{B × T × D}``, the attention computes::

        e_t  = v^T · tanh(W · h_t + b)          (energy per timestep)
        α    = softmax(e)                         (attention weights)
        c    = Σ_t  α_t · h_t                    (context vector)

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of each LSTM output vector (``D``).  For a
        bidirectional LSTM with ``hidden_dim=64`` this is ``128``.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(
        self,
        lstm_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        lstm_output : Tensor, shape (B, T, D)
            Full LSTM output sequence.
        mask : Tensor, shape (B, T), optional
            Boolean mask where ``True`` = valid timestep.

        Returns
        -------
        context : Tensor, shape (B, D)
            Weighted sum of LSTM outputs.
        weights : Tensor, shape (B, T)
            Normalised attention weights.
        """
        # (B, T, D) → (B, T, D) → (B, T, 1) → (B, T)
        energy = self.v(torch.tanh(self.W(lstm_output))).squeeze(-1)

        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))

        weights = F.softmax(energy, dim=-1)  # (B, T)

        # Weighted sum: (B, 1, T) × (B, T, D) → (B, 1, D) → (B, D)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)

        return context, weights


# DriverStateNet

class DriverStateNet(nn.Module):
    """Bi-LSTM + Temporal Attention classifier for driver state.

    Parameters
    ----------
    input_dim : int
        Feature vector dimension per frame (default 18).
    hidden_dim : int
        LSTM hidden size per direction (default 64).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    num_classes : int
        Output classes (default 3: Alert, Drowsy, Distracted).
    dropout_lstm : float
        Dropout between LSTM layers (default 0.3).
    dropout_classifier : float
        Dropout in the classifier head (default 0.4).
    bidirectional : bool
        Use bidirectional LSTM (default True).
    use_layer_norm : bool
        Apply LayerNorm on input features (default True).

    Example
    -------
    >>> model = DriverStateNet()
    >>> x = torch.randn(8, 90, 18)          # batch=8, seq=90, features=18
    >>> logits = model(x)                     # (8, 3)
    >>> attn = model.get_attention_weights(x) # (8, 90)
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout_lstm: float = 0.3,
        dropout_classifier: float = 0.4,
        bidirectional: bool = True,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm_output_dim = hidden_dim * self.num_directions  # 128

        # Input LayerNorm
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Temporal Attention
        self.attention = TemporalAttention(self.lstm_output_dim)

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_classifier),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_classifier),
            nn.Linear(64, num_classes),
        )

        self._init_weights()
        self._log_param_count()

    # Weight initialisation

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers; orthogonal for LSTM."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Set forget-gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _log_param_count(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "DriverStateNet: %s total params (%s trainable)",
            f"{total:,}",
            f"{trainable:,}",
        )

    # Forward pass

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, input_dim)
            Batch of feature sequences.
        mask : Tensor, shape (B, T), optional
            Boolean mask (True = valid).

        Returns
        -------
        logits : Tensor, shape (B, num_classes)
            Raw (un-softmaxed) class scores.
        """
        # Input normalisation
        if self.use_layer_norm:
            x = self.layer_norm(x)  # (B, T, D)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)

        # Attention pooling
        context, _ = self.attention(lstm_out, mask)  # (B, hidden*2)

        # Classification
        logits = self.classifier(context)  # (B, num_classes)
        return logits

    # Interpretability

    @torch.no_grad()
    def get_attention_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return temporal attention weights without computing gradients.

        Parameters
        ----------
        x : Tensor, shape (B, T, input_dim)
        mask : Tensor, shape (B, T), optional

        Returns
        -------
        weights : Tensor, shape (B, T)
            Normalised attention weights per timestep.
        """
        was_training = self.training
        self.eval()

        if self.use_layer_norm:
            x = self.layer_norm(x)

        lstm_out, _ = self.lstm(x)
        _, weights = self.attention(lstm_out, mask)

        if was_training:
            self.train()
        return weights

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
