"""Predict affinities for the test rows in manifest_outputtest.csv."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.pipeline import (
    AffinityDataset,
    EndToEndAffinityModel,
    affinity_collate,
    move_batch_to_device,
)


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "predict_data" / "manifest_outputtest_wuxi.csv"
MODEL_PATH = ROOT / "models" / "best_model.pt"
OUTPUT_PATH = ROOT / "results" / "output_wuxi_test.csv"
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    if checkpoint.get("ligand_encoder") != "maccs166_transformer_tokenmax_v2":
        raise ValueError("Checkpoint is not a MACCS Transformer model")
    if checkpoint.get("model_format") != "variable_states_v1" or checkpoint.get("max_states") != 10:
        raise ValueError("Checkpoint is not compatible with the variable-state model")

    model = EndToEndAffinityModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = AffinityDataset(args.manifest_path, split="test")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=affinity_collate,
    )

    predictions = []
    sample_ids = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, DEVICE)
            output = model(
                batch["fingerprint"],
                batch["patch_features"],
                batch["patch_valid_mask"],
                batch["state_probs"],
                batch["state_valid_mask"],
            )
            sample_ids.extend(batch["sample_id"])
            predictions.extend(output["affinity"].cpu().tolist())

    if len(predictions) != len(dataset) or sample_ids != [row["sample_id"] for row in dataset.rows]:
        raise RuntimeError("Prediction order does not match the manifest")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("protein_id", "ligand_smiles", "affinity"))
        for row, prediction in zip(dataset.rows, predictions):
            writer.writerow((row["protein_id"], row["ligand_smiles"], prediction))

    print(f"Predicted {len(predictions)} samples; results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
