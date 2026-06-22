"""Training loop for the Dual-Patch model."""

from __future__ import annotations

import json
import time
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.hybrid_model import DualStreamPatchTST
from preprocessing import HybridDataset, load_hybrid_data


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.7) -> None:
        super().__init__()
        self.mse_weight = mse_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = nn.functional.mse_loss(prediction, target)
        mae = nn.functional.l1_loss(prediction, target)
        return self.mse_weight * mse + (1 - self.mse_weight) * mae


def create_model(config: dict, num_transformers: int) -> DualStreamPatchTST:
    model_config = Namespace(**config)
    return DualStreamPatchTST(model_config, num_transformers)


@torch.no_grad()
def validation_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    losses = []
    for inputs, targets in loader:
        prediction = model(inputs.to(device))
        losses.append(loss_fn(prediction, targets.to(device)).item())
    return sum(losses) / len(losses)


def train_model(config: dict) -> dict:
    data = load_hybrid_data(config["data_path"], config["seq_len"], config["pred_len"])
    train_set = HybridDataset(data.train, config["seq_len"], config["pred_len"])
    val_set = HybridDataset(data.val, config["seq_len"], config["pred_len"])
    if not train_set or not val_set:
        raise ValueError("Training or validation split contains no complete windows")

    loader_options = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config, len(data.load_columns)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    loss_fn = CombinedLoss()
    checkpoint = Path(config["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []

    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Device: {device}")
    print(f"Train/validation windows: {len(train_set)}/{len(val_set)}")
    print(f"Model parameters: {parameters:,}")

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        start = time.perf_counter()
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}"):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs.to(device))
            loss = loss_fn(prediction, targets.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = validation_loss(model, val_loader, loss_fn, device)
        elapsed = time.perf_counter() - start
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f} time={elapsed:.2f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {key: value for key, value in config.items() if key not in {"mode"}},
                    "num_transformers": len(data.load_columns),
                    "load_columns": data.load_columns,
                    "scaler_mean": data.load_scaler.mean_,
                    "scaler_scale": data.load_scaler.scale_,
                    "best_val_loss": best_loss,
                },
                checkpoint,
            )

    history_path = checkpoint.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {checkpoint} (validation loss {best_loss:.6f})")
    return {"best_val_loss": best_loss, "checkpoint": str(checkpoint)}
