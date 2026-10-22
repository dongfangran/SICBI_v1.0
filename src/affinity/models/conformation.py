"""Contextualize variable-length sets of protein conformations."""

import torch
from torch import nn


class ConformationBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, state_mask: torch.Tensor):
        # Pre-normalize and use [states, batch, model_dim] for MultiheadAttention.
        normalized = self.attention_norm(states).transpose(0, 1)
        # Self-attention uses the same states as Q/K/V and ignores padded keys/values.
        attended, _ = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=~state_mask,
            need_weights=False,
        )
        # Residual attention and feed-forward updates contextualize each real state.
        states = states + self.dropout(attended.transpose(0, 1))
        states = states + self.dropout(self.ffn(self.ffn_norm(states)))
        return states.masked_fill(~state_mask.unsqueeze(-1), 0)


class ConformationEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        # No positional embeddings are added: conformations are treated as a set.
        self.layers = nn.ModuleList(
            [ConformationBlock(model_dim, num_heads, ffn_dim, dropout)
             for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, states: torch.Tensor, state_mask: torch.Tensor):
        if states.ndim != 3 or state_mask.shape != states.shape[:2]:
            raise ValueError("state_mask must have shape [batch, states]")
        state_mask = state_mask.to(dtype=torch.bool)
        if not torch.all(state_mask.any(dim=1)):
            raise ValueError("Every sample needs at least one real state")
        # Clear padded states before encoding; each block and the final norm also mask them.
        states = states.masked_fill(~state_mask.unsqueeze(-1), 0)
        for layer in self.layers:
            states = layer(states, state_mask)
        return self.output_norm(states).masked_fill(~state_mask.unsqueeze(-1), 0)

