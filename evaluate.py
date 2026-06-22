"""Checkpoint loading and test-set evaluation for Dual-Patch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from preprocessing import HybridDataset, load_hybrid_data
from train import create_model


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_flat = actual.reshape(-1)
    predicted_flat = predicted.reshape(-1)
    nonzero = np.abs(actual_flat) > 1e-8
    mape = (
        np.mean(np.abs((actual_flat[nonzero] - predicted_flat[nonzero]) / actual_flat[nonzero]))
        if nonzero.any()
        else float("nan")
    )
    return {
        "mse": float(mean_squared_error(actual_flat, predicted_flat)),
        "mae": float(mean_absolute_error(actual_flat, predicted_flat)),
        "mape_percent": float(mape * 100),
        "r2": float(r2_score(actual_flat, predicted_flat)),
    }


def predict_mode(config: dict) -> dict[str, float]:
    checkpoint_path = Path(config["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run training first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = saved["model_config"]
    model_config["data_path"] = config["data_path"]
    data = load_hybrid_data(model_config["data_path"], model_config["seq_len"], model_config["pred_len"])
    if data.load_columns != saved["load_columns"]:
        raise ValueError("Dataset load columns do not match the checkpoint")
    if not np.allclose(data.load_scaler.mean_, saved["scaler_mean"]) or not np.allclose(
        data.load_scaler.scale_, saved["scaler_scale"]
    ):
        raise ValueError("Dataset scaling statistics do not match the checkpoint")

    dataset = HybridDataset(data.test, model_config["seq_len"], model_config["pred_len"])
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    model = create_model(model_config, saved["num_transformers"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for inputs, actual in loader:
            predictions.append(model(inputs.to(device)).cpu().numpy())
            targets.append(actual.numpy())

    predicted_scaled = np.concatenate(predictions)
    actual_scaled = np.concatenate(targets)
    shape = predicted_scaled.shape
    predicted = data.load_scaler.inverse_transform(predicted_scaled.reshape(-1, shape[-1])).reshape(shape)
    actual = data.load_scaler.inverse_transform(actual_scaled.reshape(-1, shape[-1])).reshape(shape)
    metrics = regression_metrics(actual, predicted)

    output = checkpoint_path.with_suffix(".metrics.json")
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Metrics written to: {output}")
    return metrics
