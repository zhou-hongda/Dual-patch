"""Reproduce the published Dual-Patch training and prediction workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "representative_data_with_weather.csv"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "dual_patch.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reproduce", "train", "predict", "validate"), default="reproduce")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command(args: argparse.Namespace, mode: str) -> list[str]:
    return [
        args.python,
        str(ROOT / "main.py"),
        "--mode", mode,
        "--data-path", str(args.data_path),
        "--checkpoint", str(args.checkpoint),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
    ]


def validate(args: argparse.Namespace) -> int:
    required = [
        ROOT / "main.py",
        ROOT / "train.py",
        ROOT / "evaluate.py",
        ROOT / "preprocessing.py",
        ROOT / "model" / "hybrid_model.py",
        ROOT / "config" / "representative_transformers.txt",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        print("Missing code files:", ", ".join(missing))
        return 1
    compiled = subprocess.run(
        [args.python, "-m", "py_compile", *(str(path) for path in required[:-1])],
        cwd=ROOT,
        check=False,
    )
    if compiled.returncode:
        print("Python compilation failed.")
        return compiled.returncode
    if not args.data_path.is_file():
        print(f"Code validation passed; dataset is not present at {args.data_path}")
        print("Download or prepare the dataset before training (see README.md).")
        return 0
    print(f"Validation passed, including dataset: {args.data_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "validate":
        return validate(args)

    stages = ("train", "predict") if args.mode == "reproduce" else (args.mode,)
    run_dir = ROOT / "results" / "experiment_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for stage in stages:
        stage_command = command(args, stage)
        print(subprocess.list2cmdline(stage_command))
        if args.dry_run:
            records.append({"stage": stage, "status": "dry-run"})
            continue
        start = time.perf_counter()
        completed = subprocess.run(stage_command, cwd=ROOT, check=False)
        records.append({
            "stage": stage,
            "return_code": completed.returncode,
            "seconds": round(time.perf_counter() - start, 3),
        })
        (run_dir / "run.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
