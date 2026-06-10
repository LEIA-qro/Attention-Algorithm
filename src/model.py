"""DriverStateNet: bi-LSTM with temporal attention over 18-D feature sequences.

LayerNorm -> bi-LSTM -> additive attention pooling -> MLP head.
forward() returns raw (B, 3) logits.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DriverStateNet", "TemporalAttention"]

logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """additive (bahdanau) attention pooling over the lstm time axis"""

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
        """return (context (B,D), weights (B,T)); mask True = valid timestep"""
        # energy per timestep: (B,T,D) -> (B,T)
        energy = self.v(torch.tanh(self.W(lstm_output))).squeeze(-1)

        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))

        weights = F.softmax(energy, dim=-1)  # (B, T)

        # weighted sum over time -> (B,D)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)

        return context, weights


class DriverStateNet(nn.Module):
    """bi-lstm + temporal attention classifier for driver state (alert/drowsy/distracted)"""

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
        self.lstm_output_dim = hidden_dim * self.num_directions

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.attention = TemporalAttention(self.lstm_output_dim)

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

    def _init_weights(self) -> None:
        # xavier for linear, orthogonal for lstm recurrent weights
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # forget-gate bias = 1
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """return raw class logits (B, num_classes)"""
        if self.use_layer_norm:
            x = self.layer_norm(x)  # (B, T, D)

        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)
        context, _ = self.attention(lstm_out, mask)  # (B, hidden*2)
        logits = self.classifier(context)  # (B, num_classes)
        return logits

    @torch.no_grad()
    def get_attention_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """return temporal attention weights (B, T), no grad"""
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
