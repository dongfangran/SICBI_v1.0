import math

import torch
from torch import nn

from .conformation import ConformationEncoder
from .patch_pooling import PatchPooling


class AffinityModel(nn.Module):
    """Ligand-conditioned affinity prediction over up to ten protein states."""

    def __init__(
        self,
        patch_dim: int = 16,
        ligand_dim: int = 166,
        ligand_input_dim: int = 166,
        model_dim: int = 128,
        projection_dim: int = 64,
        num_queries: int = 4,
        patch_num_heads: int = 8,
        patch_ffn_dim: int = 512,
        conformation_num_heads: int = 8,
        conformation_ffn_dim: int = 512,
        state_layers: int = 1,
        num_states: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.ligand_dim = ligand_dim
        self.ligand_input_dim = ligand_input_dim
        self.num_states = num_states

        # Preserve the ligand vector when input and internal widths already match.
        self.ligand_projection = (
            nn.Identity()
            if ligand_input_dim == ligand_dim
            else nn.Linear(ligand_input_dim, ligand_dim)
        )
        # Use separate attention/FFN settings for patch pooling and conformation encoding.
        self.patch_pooling = PatchPooling(
            patch_dim=patch_dim,
            ligand_dim=ligand_dim,
            model_dim=model_dim,
            projection_dim=projection_dim,
            num_queries=num_queries,
            num_heads=patch_num_heads,
            ffn_dim=patch_ffn_dim,
            dropout=dropout,
        )
        self.conformation_encoder = ConformationEncoder(
            model_dim=model_dim,
            num_heads=conformation_num_heads,
            ffn_dim=conformation_ffn_dim,
            num_layers=state_layers,
            dropout=dropout,
        )
        # Regress one score from each contextual state concatenated with the ligand.
        self.state_affinity = nn.Sequential(
            nn.Linear(model_dim + ligand_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        patch_mask: torch.Tensor,
        state_probs: torch.Tensor,
        ligand_vector: torch.Tensor,
        state_valid_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        if patch_features.ndim != 4:
            raise ValueError("patch_features must have shape [batch, states, patches, patch_dim]")
        batch_size, num_states, _, patch_dim = patch_features.shape
        if not 1 <= num_states <= self.num_states:
            raise ValueError(f"Expected 1 to {self.num_states} protein states, got {num_states}")
        if patch_dim != self.patch_dim:
            raise ValueError(f"Expected patch dimension {self.patch_dim}, got {patch_dim}")
        if patch_mask.shape != patch_features.shape[:3]:
            raise ValueError("patch_mask must match the first three patch_features dimensions")
        if state_probs.shape != (batch_size, num_states):
            raise ValueError("state_probs must have shape [batch, states]")
        if state_valid_mask.shape != (batch_size, num_states):
            raise ValueError("state_valid_mask must have shape [batch, states]")
        if ligand_vector.shape != (batch_size, self.ligand_input_dim):
            raise ValueError(f"ligand_vector must have shape [batch, {self.ligand_input_dim}]")

        # Real states require patches; padded states must carry zero probability.
        state_valid_mask = state_valid_mask.bool()
        if not torch.all(state_valid_mask.any(dim=1)):
            raise ValueError("Every sample must have at least one protein state")
        if not torch.equal(patch_mask.bool().any(dim=-1), state_valid_mask):
            raise ValueError("Every real state needs patches; padded states cannot have patches")
        if not torch.isfinite(state_probs).all() or torch.any(state_probs < 0):
            raise ValueError("State probabilities must be finite and non-negative")
        if torch.any(state_probs.masked_select(~state_valid_mask) != 0):
            raise ValueError("Padded states must have zero probability")

        # Retrieve ligand-relevant patches before modeling relations between states.
        ligand_vector = self.ligand_projection(ligand_vector)
        state_embeddings, patch_attention = self.patch_pooling(
            patch_features,
            ligand_vector,
            patch_mask,
            state_valid_mask,
            return_attention=return_attention,
        )
        contextual_states = self.conformation_encoder(state_embeddings, state_valid_mask)
        # Share the same ligand vector across all conformations of one sample.
        ligand_per_state = ligand_vector[:, None, :].expand(-1, num_states, -1)
        state_scores = self.state_affinity(
            torch.cat([contextual_states, ligand_per_state], dim=-1)
        ).squeeze(-1).masked_fill(~state_valid_mask, 0)

        # Normalize supplied state priors independently for each sample.
        probabilities = state_probs.to(state_scores.dtype)
        probability_sum = probabilities.sum(dim=1, keepdim=True)
        if torch.any(probability_sum <= 0):
            raise ValueError("Every sample must have positive probability mass")
        probabilities = probabilities / probability_sum
        # Exclude padded and zero-probability states from the logarithmic ensemble.
        positive = state_valid_mask & (probabilities > 0)
        log_probabilities = torch.full_like(probabilities, float("-inf"))
        log_probabilities = torch.where(
            positive,
            torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)),
            log_probabilities,
        )

        # Compute log10(sum(p_i * 10**score_i)) via logsumexp for numerical stability.
        ln10 = state_scores.new_tensor(math.log(10.0))
        ensemble_logits = log_probabilities + ln10 * state_scores
        affinity = torch.logsumexp(ensemble_logits, dim=1) / ln10
        # Diagnostic contributions depend on both state priors and predicted scores.
        state_contributions = torch.softmax(ensemble_logits, dim=1)

        output = {
            "affinity": affinity,
            "state_scores": state_scores,
            "state_contributions": state_contributions,
            "state_embeddings": contextual_states,
        }
        if return_attention:
            output["patch_attention"] = patch_attention
        return output

