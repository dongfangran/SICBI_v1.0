"""A compact Transformer encoder for fixed-position MACCS fingerprints."""

import torch
from torch import nn

from .fingerprint import MACCS_BITS


class LigandTransformer(nn.Module):
    """Map [batch, 166] MACCS bits to [batch, 166] ligand vectors."""

    def __init__(
        self,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim <= 0 or num_heads <= 0 or model_dim % num_heads != 0:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")

        # The output stays 166-dimensional even when the internal model width changes.
        self.output_dim = MACCS_BITS
        # Embed the bit value and its chemical-key identity; zero bits are not padding.
        self.value_embedding = nn.Embedding(2, model_dim) 
        self.bit_embedding = nn.Embedding(MACCS_BITS, model_dim) 
        self.input_dropout = nn.Dropout(dropout) 

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.token_head = nn.Linear(model_dim, 2)

    def forward(self, fingerprint: torch.Tensor) -> torch.Tensor:
        if fingerprint.ndim != 2 or fingerprint.shape[1] != MACCS_BITS:
            raise ValueError("fingerprint must have shape [batch, 166]")
        if fingerprint.dtype != torch.long:
            raise TypeError("fingerprint must contain integer MACCS bits")
        if torch.any((fingerprint != 0) & (fingerprint != 1)):
            raise ValueError("fingerprint values must be zero or one")

        # Represent each MACCS key as one token: [batch, 166, model_dim].
        bit_positions = torch.arange(MACCS_BITS, device=fingerprint.device)
        embedded = self.value_embedding(fingerprint) + self.bit_embedding(bit_positions)
        encoded = self.encoder(self.input_dropout(embedded))
        # Pool the two scores of each token, preserving all 166 fingerprint positions.
        return self.token_head(self.output_norm(encoded)).amax(dim=-1)  #[B, 166, 2] -> [B, 166]