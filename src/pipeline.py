"""Shared data pipeline and end-to-end model for variable protein state counts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .affinity import AffinityModel
from .Ligand_transformer import MACCS_BITS, LigandTransformer, smiles_to_maccs


# Upper bound on real conformations; each patch has 16 precomputed features.
MAX_STATES = 10
PATCH_DIM = 16
REQUIRED_COLUMNS = {
    "sample_id", "protein_id", "protein_path", "ligand_id",
    "ligand_smiles", "state_prob_path", "affinity", "split",
}


def read_manifest(
    manifest_path: Union[str, Path], split: Optional[str] = None
) -> List[dict]:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Manifest has no header")
        # Validate the shared manifest schema used by all entry points.
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        # Filter the requested split while preserving the original row order.
        rows = [row for row in reader if split is None or row["split"] == split]
    if not rows:
        raise ValueError(f"No samples found for split {split!r}")
    return rows


def _load_protein(path: Path) -> tuple[List[torch.Tensor], np.ndarray]:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != PATCH_DIM + 1 or len(array) == 0:
        raise ValueError(f"{path} must have nonempty shape [N, {PATCH_DIM + 1}]")
    if not np.isfinite(array).all():
        raise ValueError(f"{path} contains NaN or Inf")

    # Column 0 stores state IDs; the remaining columns store patch descriptors.
    raw_ids = array[:, 0]
    if not np.allclose(raw_ids, np.rint(raw_ids)):
        raise ValueError(f"State IDs in {path} must be integers")
    integer_ids = np.rint(raw_ids).astype(np.int64)
    state_ids = np.unique(integer_ids)
    if not 1 <= len(state_ids) <= MAX_STATES:
        raise ValueError(f"{path} must contain 1 to {MAX_STATES} states")

    # Group patches by sorted state ID; each tensor has shape [patches, 16].
    states = [
        torch.from_numpy(array[integer_ids == state_id, 1:].astype(np.float32, copy=True))
        for state_id in state_ids
    ]
    return states, state_ids


def _load_state_probabilities(path: Path, state_ids: np.ndarray) -> torch.Tensor:
    # The probability table contains [state ID, probability] for each real state.
    table = np.load(path, allow_pickle=False)
    if table.ndim != 2 or table.shape != (len(state_ids), 2):
        raise ValueError(f"{path} must have shape [{len(state_ids)}, 2]")
    if not np.isfinite(table).all():
        raise ValueError(f"{path} contains NaN or Inf")
    raw_ids = table[:, 0]
    if not np.allclose(raw_ids, np.rint(raw_ids)):
        raise ValueError(f"State IDs in {path} must be integers")
    probability_ids = np.rint(raw_ids).astype(np.int64)
    if not np.array_equal(np.sort(probability_ids), state_ids):
        raise ValueError(f"State IDs in {path} do not match the protein features")
    # Align probabilities with the state order used by the protein tensors.
    probability_map = dict(zip(probability_ids.tolist(), table[:, 1].tolist()))
    probabilities = np.asarray(
        [probability_map[state_id] for state_id in state_ids], dtype=np.float32
    )
    if (probabilities < 0).any() or probabilities.sum() <= 0:
        raise ValueError(f"Invalid state probabilities in {path}")
    # Normalize the probability mass before batching.
    probabilities /= probabilities.sum()
    return torch.from_numpy(probabilities)


class AffinityDataset(Dataset):
    """Load MACCS bits, protein patches, state probabilities, and affinity."""

    def __init__(
        self, manifest_path: Union[str, Path], split: Optional[str] = None
    ) -> None:
        self.manifest_path = Path(manifest_path)
        # Resolve manifest-relative NPY paths from this directory.
        self.data_dir = self.manifest_path.parent
        self.rows = read_manifest(self.manifest_path, split)
        self._fingerprint_cache: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        smiles = row["ligand_smiles"].strip()
        if not smiles:
            raise ValueError(f"Sample {row['sample_id']} has an empty SMILES")
        # Reuse fingerprints for repeated SMILES; protein arrays are loaded per sample.
        if smiles not in self._fingerprint_cache:
            try:
                self._fingerprint_cache[smiles] = smiles_to_maccs(smiles)
            except ValueError as exc:
                raise ValueError(f"Invalid ligand in sample {row['sample_id']}") from exc

        protein_path = self.data_dir / row["protein_path"]
        probability_path = self.data_dir / row["state_prob_path"]
        if not protein_path.is_file() or not probability_path.is_file():
            raise FileNotFoundError(f"Missing protein data for {row['sample_id']}")
        states, state_ids = _load_protein(protein_path)
        state_probs = _load_state_probabilities(probability_path, state_ids)

        return {
            "sample_id": row["sample_id"],
            "fingerprint": self._fingerprint_cache[smiles],
            "patch_states": states,
            "state_probs": state_probs,
            # Prediction manifests still need a numeric placeholder label, such as 0.0.
            "affinity": torch.tensor(float(row["affinity"]), dtype=torch.float32),
        }


def affinity_collate(samples: List[Dict[str, object]]) -> Dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(samples)
    # Pad to the largest state and patch counts in this batch, not the whole dataset.
    max_states = max(len(sample["patch_states"]) for sample in samples)
    max_patches = max(
        state.shape[0] for sample in samples for state in sample["patch_states"]
    )
    patch_features = torch.zeros(
        batch_size, max_states, max_patches, PATCH_DIM, dtype=torch.float32
    )
    # True marks real data; padded patches/states and their probabilities stay zero.
    patch_valid_mask = torch.zeros(batch_size, max_states, max_patches, dtype=torch.bool)
    state_valid_mask = torch.zeros(batch_size, max_states, dtype=torch.bool)
    state_probs = torch.zeros(batch_size, max_states, dtype=torch.float32)

    for batch_index, sample in enumerate(samples):
        states = sample["patch_states"]
        num_states = len(states)
        if not 1 <= num_states <= MAX_STATES or sample["state_probs"].shape != (num_states,):
            raise ValueError(f"Every sample must have 1 to {MAX_STATES} aligned states")
        state_valid_mask[batch_index, :num_states] = True
        state_probs[batch_index, :num_states] = sample["state_probs"]
        for state_index, state in enumerate(states):
            num_patches = state.shape[0]
            if num_patches == 0 or state.shape[1] != PATCH_DIM:
                raise ValueError("Every real state must contain 16-dimensional patches")
            patch_features[batch_index, state_index, :num_patches] = state
            patch_valid_mask[batch_index, state_index, :num_patches] = True

    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        "fingerprint": torch.stack([sample["fingerprint"] for sample in samples]),
        "patch_features": patch_features,
        "patch_valid_mask": patch_valid_mask,
        "state_valid_mask": state_valid_mask,
        "state_probs": state_probs,
        "affinity": torch.stack([sample["affinity"] for sample in samples]),
    }


# Move model tensors together while keeping sample IDs as Python metadata.
def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> dict:
    return {
        "sample_id": batch["sample_id"],
        "fingerprint": batch["fingerprint"].to(device),
        "patch_features": batch["patch_features"].to(device),
        "patch_valid_mask": batch["patch_valid_mask"].to(device),
        "state_valid_mask": batch["state_valid_mask"].to(device),
        "state_probs": batch["state_probs"].to(device),
        "affinity": batch["affinity"].to(device),
    }


class EndToEndAffinityModel(nn.Module):
    """MACCS Transformer followed by the multi-conformation AffinityModel."""

    def __init__(self) -> None:
        super().__init__()
        self.ligand_encoder = LigandTransformer()
        # Pass the ligand encoder output width to both affinity-model ligand interfaces.
        self.affinity_model = AffinityModel(
            ligand_dim=self.ligand_encoder.output_dim,
            ligand_input_dim=self.ligand_encoder.output_dim,
            num_states=MAX_STATES,
        )

    def forward(
        self,
        fingerprint: torch.Tensor,
        patch_features: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        state_probs: torch.Tensor,
        state_valid_mask: torch.Tensor,
    ) -> dict:
        if fingerprint.ndim != 2 or fingerprint.shape[1] != MACCS_BITS:
            raise ValueError("fingerprint must have shape [batch, 166]")
        batch_size = fingerprint.shape[0]
        if patch_features.ndim != 4:
            raise ValueError("patch_features must have shape [batch, states, patches, 16]")
        if patch_features.shape[0] != batch_size or not 1 <= patch_features.shape[1] <= MAX_STATES:
            raise ValueError("Protein batch size or state count is invalid")
        if patch_features.shape[3] != PATCH_DIM:
            raise ValueError("Protein patch feature dimension must be 16")
        if patch_valid_mask.shape != patch_features.shape[:3]:
            raise ValueError("patch_valid_mask shape does not match patch_features")
        if state_valid_mask.shape != patch_features.shape[:2]:
            raise ValueError("state_valid_mask shape does not match patch_features")
        if state_probs.shape != patch_features.shape[:2]:
            raise ValueError("state_probs shape does not match patch_features")

        # Encode [batch, 166] bits, then combine the ligand vector with protein states.
        ligand_vector = self.ligand_encoder(fingerprint)
        return self.affinity_model(
            patch_features, patch_valid_mask, state_probs, ligand_vector,
            state_valid_mask,
        )
