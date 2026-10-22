"""End-to-end training with warmup, cosine decay, and early stopping."""

import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.pipeline import (
    AffinityDataset,
    EndToEndAffinityModel,
    affinity_collate,
    move_batch_to_device,
)


# Parameters
ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "train_data" / "demo_manifest.csv"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "demo_logs"
MODEL_PATH = MODEL_DIR / "demo_best_model.pt"
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 6
SEED = int(os.environ.get("SICBI_SEED", "42"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scheduler) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, DEVICE)
        optimizer.zero_grad()
        output = model(
            batch["fingerprint"],
            batch["patch_features"],
            batch["patch_valid_mask"],
            batch["state_probs"],
            batch["state_valid_mask"],
        )
        loss = criterion(output["affinity"], batch["affinity"])
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = batch["affinity"].shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def validate(model, loader, criterion) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

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
            loss = criterion(output["affinity"], batch["affinity"])

            batch_size = batch["affinity"].shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def make_lr_scheduler(optimizer, total_steps: int) -> LambdaLR:
    """Linearly warm up to LEARNING_RATE, then decay to MIN_LEARNING_RATE."""
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    decay_steps = max(1, total_steps - warmup_steps)
    minimum_scale = MIN_LEARNING_RATE / LEARNING_RATE

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = min((step - warmup_steps) / decay_steps, 1.0)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_scale + (1.0 - minimum_scale) * cosine_scale

    return LambdaLR(optimizer, lr_lambda=lr_multiplier)


def main() -> None:
    set_random_seed(SEED)
    print(f"Training seed: {SEED}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = AffinityDataset(MANIFEST_PATH, split="train")
    validation_dataset = AffinityDataset(MANIFEST_PATH, split="valid")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=affinity_collate,
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=affinity_collate,
    )

    model = EndToEndAffinityModel().to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = EPOCHS * len(train_loader)
    scheduler = make_lr_scheduler(optimizer, total_steps)

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    validation_losses = []

    for epoch in range(1, EPOCHS + 1):
        training_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler
        )
        validation_loss = validate(model, validation_loader, criterion)
        train_losses.append(training_loss)
        validation_losses.append(validation_loss)
        current_learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"lr={current_learning_rate:.2e} | "
            f"train_loss={training_loss:.6f} | "
            f"validation_loss={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "ligand_encoder": "maccs166_transformer_tokenmax_v2",
                    "model_format": "variable_states_v1",
                    "max_states": 10,
                    "validation_loss": validation_loss,
                    "epoch": epoch,
                    "seed": SEED,
                },
                MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    "Early stopping: validation loss did not improve for "
                    f"{EARLY_STOPPING_PATIENCE} consecutive epochs. "
                    f"Keeping the best checkpoint with validation_loss="
                    f"{best_validation_loss:.6f}."
                )
                break

    print(f"Best model saved to: {MODEL_PATH}")

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, validation_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOG_DIR / "loss_curve_transformer.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()


