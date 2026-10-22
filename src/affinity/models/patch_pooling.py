"""Pool protein patches with queries conditioned on the ligand."""

import torch
from torch import nn


class PatchPooling(nn.Module):
    def __init__(
        self,
        patch_dim: int,
        ligand_dim: int,
        model_dim: int,
        projection_dim: int,
        num_queries: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_dim = patch_dim
        self.ligand_dim = ligand_dim
        self.model_dim = model_dim
        self.num_queries = num_queries

        # Project patch descriptors to the attention width (default: 16 -> 64 -> 128).
        self.patch_projection = nn.Sequential(
            nn.Linear(patch_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        # Combine learned query templates with a ligand-dependent query offset.
        self.base_queries = nn.Parameter(torch.empty(num_queries, model_dim))
        self.ligand_to_queries = nn.Linear(ligand_dim, num_queries * model_dim)
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.attention_norm = nn.LayerNorm(model_dim)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.query_gate = nn.Linear(model_dim, 1)
        nn.init.normal_(self.base_queries, std=0.02)

    def forward(
        self,
        patch_features: torch.Tensor,
        ligand_vector: torch.Tensor,
        patch_mask: torch.Tensor,
        state_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        if patch_features.ndim != 4:
            raise ValueError("patch_features must have shape [batch, states, patches, patch_dim]")
        batch_size, num_states, num_patches, patch_dim = patch_features.shape
        if patch_dim != self.patch_dim or num_patches < 1:
            raise ValueError("Invalid protein patch dimensions")
        if patch_mask.shape != patch_features.shape[:3]:
            raise ValueError("patch_mask shape does not match patch_features")
        if state_mask.shape != (batch_size, num_states):
            raise ValueError("state_mask must have shape [batch, states]")
        if ligand_vector.shape != (batch_size, self.ligand_dim):
            raise ValueError("ligand_vector shape is incompatible")

        patch_mask = patch_mask.to(dtype=torch.bool)
        state_mask = state_mask.to(dtype=torch.bool)
        if not torch.equal(patch_mask.any(dim=-1), state_mask):
            raise ValueError("Every real state needs patches; padded states cannot have patches")

        # Process only real states to avoid attention over entirely masked patch sets.
        valid_indices = state_mask.reshape(-1).nonzero(as_tuple=True)[0]
        flat_patches = patch_features.reshape(batch_size * num_states, num_patches, patch_dim)
        patches = self.patch_projection(flat_patches.index_select(0, valid_indices))
        valid_patches = patch_mask.reshape(batch_size * num_states, num_patches).index_select(
            0, valid_indices
        )
        # Broadcast each ligand to its states; queries are projected to model_dim.
        ligand = ligand_vector[:, None, :].expand(-1, num_states, -1)
        ligand = ligand.reshape(batch_size * num_states, self.ligand_dim)
        ligand = ligand.index_select(0, valid_indices)
        queries = self.ligand_to_queries(ligand).reshape(-1, self.num_queries, self.model_dim)
        queries = queries + self.base_queries.unsqueeze(0)

        # Ligand-conditioned queries attend to patch keys/values; masks exclude padding.
        attended, attention = self.attention(
            queries.transpose(0, 1),
            patches.transpose(0, 1),
            patches.transpose(0, 1),
            key_padding_mask=~valid_patches,
            need_weights=return_attention,
        )
        queries = self.attention_norm(queries + self.dropout(attended.transpose(0, 1)))
        queries = self.ffn_norm(queries + self.dropout(self.ffn(queries)))
        # Learn how to combine multiple query outputs into one vector per state.
        gates = torch.softmax(self.query_gate(queries).squeeze(-1), dim=1)
        pooled = torch.sum(gates.unsqueeze(-1) * queries, dim=1)

        # Restore the batch/state layout, leaving padded state embeddings as zero.
        all_states = pooled.new_zeros(batch_size * num_states, self.model_dim)
        all_states = all_states.index_copy(0, valid_indices, pooled)
        all_states = all_states.reshape(batch_size, num_states, self.model_dim)

        # Optionally restore patch attention maps for inspection, including padded slots.
        if return_attention:
            all_attention = attention.new_zeros(
                batch_size * num_states, self.num_queries, num_patches
            )
            all_attention = all_attention.index_copy(0, valid_indices, attention)
            attention = all_attention.reshape(
                batch_size, num_states, self.num_queries, num_patches
            )
        return all_states, attention

