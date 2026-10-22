"""Evaluate test predictions and draw a true-vs-predicted scatter plot."""

import argparse
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


# Parameters
ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "train_data" / "manifest.csv"
MODEL_PATH = ROOT / "models" / "best_model.pt"
PLOT_PATH = ROOT / "results" / "test_scatter.png"
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    if checkpoint.get("ligand_encoder") != "maccs166_transformer_tokenmax_v2":
        raise ValueError("Checkpoint is not a MACCS Transformer model")

    if checkpoint.get("model_format") != "variable_states_v1" or checkpoint.get("max_states") != 10:
        raise ValueError("Checkpoint is not compatible with the variable-state model")

    model = EndToEndAffinityModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = AffinityDataset(MANIFEST_PATH, split="test")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=affinity_collate,
    )

    true_values = []
    predicted_values = []

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
            true_values.extend(batch["affinity"].cpu().tolist())
            predicted_values.extend(output["affinity"].cpu().tolist())

    true_array = np.asarray(true_values)
    predicted_array = np.asarray(predicted_values)
    mse = float(np.mean((predicted_array - true_array) ** 2))
    if np.var(true_array) == 0 or np.var(predicted_array) == 0:
        raise ValueError("r_m^2 requires non-constant true and predicted values")

    r_squared = np.corrcoef(true_array, predicted_array)[0, 1] ** 2
    k = np.sum(true_array * predicted_array) / np.sum(predicted_array ** 2)
    r0_squared = 1 - np.sum((true_array - k * predicted_array) ** 2) / np.sum(
        (true_array - np.mean(true_array)) ** 2
    )
    rm_squared = r_squared * (1 - np.sqrt(max(r_squared - r0_squared, 0.0)))

    lower = min(true_values + predicted_values)
    upper = max(true_values + predicted_values)
    plt.figure(figsize=(6, 6))
    plt.scatter(true_values, predicted_values, alpha=0.7)
    plt.plot([lower, upper], [lower, upper], "--", color="gray")
    plt.xlabel("True affinity")
    plt.ylabel("Predicted affinity")
    plt.title(rf"$r_m^2$ = {rm_squared:.4f}")
    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()

    print(f"Predicted {len(dataset)} test samples")
    print(f"r_m^2={rm_squared:.6f}")
    print(f"MSE={mse:.6f}")
    print(f"Scatter plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()

