"""Data loading for paired transformer-load and temperature time series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


@dataclass
class HybridData:
    train: tuple[np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]
    load_scaler: StandardScaler
    load_columns: list[str]


class HybridDataset(Dataset):
    """Sliding windows with load and temperature as input, and load as target."""

    def __init__(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
        seq_len: int,
        pred_len: int,
        stride: int = 1,
    ) -> None:
        load, temperature = arrays
        if load.shape != temperature.shape:
            raise ValueError(f"Load and temperature shapes differ: {load.shape} vs {temperature.shape}")
        self.inputs = np.concatenate((load, temperature), axis=1).astype(np.float32)
        self.targets = load.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride

    def __len__(self) -> int:
        windows = len(self.inputs) - self.seq_len - self.pred_len + 1
        return max(0, (windows - 1) // self.stride + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        split = start + self.seq_len
        end = split + self.pred_len
        return torch.from_numpy(self.inputs[start:split]), torch.from_numpy(self.targets[split:end])


def load_hybrid_data(
    file_path: str | Path,
    seq_len: int,
    pred_len: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> HybridData:
    """Load, validate, scale, and chronologically split the hybrid data."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. See README.md for the required CSV schema."
        )
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")

    frame = pd.read_csv(path)
    date_column = "DATETIME" if "DATETIME" in frame.columns else frame.columns[0]
    dates = pd.to_datetime(frame.pop(date_column), errors="raise")
    if not dates.is_monotonic_increasing:
        order = np.argsort(dates.to_numpy())
        frame = frame.iloc[order].reset_index(drop=True)

    load_columns = sorted(column for column in frame if not column.startswith("TEMP_"))
    if not load_columns:
        raise ValueError("No load columns found")
    temp_columns = [f"TEMP_{column}" for column in load_columns]
    missing = [column for column in temp_columns if column not in frame]
    if missing:
        raise ValueError(f"Missing paired temperature columns: {', '.join(missing)}")

    selected = frame[load_columns + temp_columns].apply(pd.to_numeric, errors="coerce")
    if selected.isna().any().any():
        bad = selected.columns[selected.isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values in columns: {', '.join(bad)}")

    load = selected[load_columns].to_numpy(dtype=np.float32)
    temperature = selected[temp_columns].to_numpy(dtype=np.float32)
    train_end = int(len(frame) * train_ratio)
    val_end = train_end + int(len(frame) * val_ratio)
    minimum = seq_len + pred_len
    if train_end < minimum or val_end - train_end < pred_len or len(frame) - val_end < pred_len:
        raise ValueError(f"Dataset is too short for seq_len={seq_len} and pred_len={pred_len}")

    load_scaler = StandardScaler().fit(load[:train_end])
    temp_scaler = StandardScaler().fit(temperature[:train_end])
    load = load_scaler.transform(load).astype(np.float32)
    temperature = temp_scaler.transform(temperature).astype(np.float32)

    def segment(start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        return load[start:end], temperature[start:end]

    return HybridData(
        train=segment(0, train_end),
        val=segment(train_end - seq_len, val_end),
        test=segment(val_end - seq_len, len(frame)),
        load_scaler=load_scaler,
        load_columns=load_columns,
    )
