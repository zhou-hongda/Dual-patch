"""Command-line entry point for Dual-Patch training and evaluation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

import evaluate
import train


PROJECT_ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate the Dual-Patch model.")
    parser.add_argument("--mode", choices=("train", "predict"), default="train")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "representative_data_with_weather.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "dual_patch.pt",
    )
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.patch_len > args.seq_len:
        raise ValueError("--patch-len cannot exceed --seq-len")
    if args.stride <= 0 or args.patch_len <= 0:
        raise ValueError("--patch-len and --stride must be positive")
    if args.d_model % args.n_heads:
        raise ValueError("--d-model must be divisible by --n-heads")
    set_seed(args.seed)
    config = vars(args)

    if args.mode == "train":
        train.train_model(config)
    else:
        evaluate.predict_mode(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
