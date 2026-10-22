"""Convert a ligand SMILES string to the 166 defined MACCS keys."""

import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys


MACCS_BITS = 166


def smiles_to_maccs(smiles: str) -> torch.Tensor:
    """Return the MACCS bits as a long tensor of shape [166].

    RDKit returns 167 positions; position zero is unused and is discarded.
    Both zero and one are meaningful fingerprint values, not padding tokens.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("Ligand SMILES must be a non-empty string")

    # Parse the SMILES before generating deterministic MACCS keys.
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise ValueError(f"Invalid ligand SMILES: {smiles!r}")

    # Discard RDKit position 0 and retain the 166 defined fingerprint bits.
    bit_string = MACCSkeys.GenMACCSKeys(molecule).ToBitString()[1:]
    if len(bit_string) != MACCS_BITS:
        raise ValueError(f"Expected {MACCS_BITS} MACCS bits, got {len(bit_string)}")
    return torch.tensor([int(bit) for bit in bit_string], dtype=torch.long)
