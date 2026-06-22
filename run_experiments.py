"""Run the submitted Dual-Patch experiments with one reproducible entry point."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
RESULTS_DIR = PROJECT_ROOT / "results"

# All models referenced in the manuscript text, figures, or result discussion.
PAPER_MODELS = [
    "PatchTST",
    "iTransformer",
    "Crossformer",
    # "TimesNet",
    "Autoformer",
    "Informer",
    "DLinear",
    "TSMixer",
    "DeepESN",
    "RVFL",
    "GRU",
    "RNN",
    "TCN",
    "CNN1D",
    "CNN-LSTM",
    "Transformer-LSTM",
    "MLP",
    "LSTM",
]

# Models reported in the manuscript's computational-efficiency table.
EFFICIENCY_MODELS = ["DLinear", "iTransformer", "PatchTST", "Crossformer", "Informer"]

PAPER_CONFIG = {
    "epochs": 15,
    "batch_size": 64,
    "learning_rate": 1e-3,
}

METRIC_PATTERNS = {
    "parameters_m": re.compile(r"Model Parameters:\s*([0-9.]+)\s*M"),
    "inference_seconds": re.compile(r"Test Inference Time:\s*([0-9.]+)\s*seconds"),
    "mse": re.compile(r"Normalized MSE:\s*([0-9.]+)"),
    "mae": re.compile(r"^\s*MAE:\s*([0-9.]+)"),
    "mape": re.compile(r"^\s*MAPE:\s*([0-9.]+)%"),
    "r2": re.compile(r"^\s*R2:\s*([+-]?[0-9.]+)"),
}
TRAIN_TIME_PATTERN = re.compile(r"Training Time:\s*([0-9.]+)\s*seconds")
VALIDATION_TIME_PATTERN = re.compile(r"Total Inference Time:\s*([0-9.]+)\s*seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, compare, or benchmark the models used in the manuscript."
    )
    parser.add_argument(
        "--mode",
        choices=["reproduce", "train", "predict", "compare", "efficiency", "validate"],
        default="reproduce",
        help="reproduce runs train, predict, and comparison stages in sequence",
    )
    parser.add_argument("--models", nargs="+", help="override the default manuscript model set")
    parser.add_argument("--epochs", type=int, default=PAPER_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=PAPER_CONFIG["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=PAPER_CONFIG["learning_rate"])
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child runs")
    parser.add_argument("--run-name", help="output folder name under results/experiment_logs")
    parser.add_argument("--skip-existing", action="store_true", help="skip training when a checkpoint exists")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failed command")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    return parser.parse_args()


def selected_models(args: argparse.Namespace) -> list[str]:
    models = args.models or (EFFICIENCY_MODELS if args.mode == "efficiency" else PAPER_MODELS)
    duplicates = [name for i, name in enumerate(models) if name in models[:i]]
    unknown = [name for name in models if name not in PAPER_MODELS]
    if duplicates:
        raise ValueError(f"Duplicate model names: {duplicates}")
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")
    return models


def paper_settings_changed(args: argparse.Namespace) -> bool:
    return (
        args.epochs != PAPER_CONFIG["epochs"]
        or args.batch_size != PAPER_CONFIG["batch_size"]
        or args.learning_rate != PAPER_CONFIG["learning_rate"]
    )


def command_for(args: argparse.Namespace, mode: str, model: str | None = None,
                models: list[str] | None = None, epochs: int | None = None) -> list[str]:
    command = [args.python, str(MAIN_SCRIPT), "--mode", mode]
    if model is not None:
        command.extend(["--model", model])
    if models is not None:
        command.extend(["--compare_models", *models])
    command.extend([
        "--epochs", str(args.epochs if epochs is None else epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
    ])
    return command


def parse_output_line(record: dict, line: str) -> None:
    train_match = TRAIN_TIME_PATTERN.search(line)
    if train_match:
        times = record.setdefault("_train_times", [])
        times.append(float(train_match.group(1)))
        record["epochs_completed"] = len(times)
        record["train_seconds_total"] = round(sum(times), 4)
        record["train_seconds_mean"] = round(sum(times) / len(times), 4)

    validation_match = VALIDATION_TIME_PATTERN.search(line)
    if validation_match:
        times = record.setdefault("_validation_times", [])
        times.append(float(validation_match.group(1)))
        record["validation_seconds_mean"] = round(sum(times) / len(times), 4)

    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(line)
        if match:
            record[key] = float(match.group(1))


def run_command(command: list[str], log_path: Path, stage: str,
                model: str, dry_run: bool) -> dict:
    record = {
        "model": model,
        "stage": stage,
        "status": "dry-run" if dry_run else "running",
        "wall_seconds": 0.0,
        "command": subprocess.list2cmdline(command),
    }
    print(f"\n[{stage.upper()}] {model}")
    print(record["command"])
    if dry_run:
        return record

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            parse_output_line(record, line)
        return_code = process.wait()

    record["wall_seconds"] = round(time.perf_counter() - start, 4)
    record["return_code"] = return_code
    record["status"] = "success" if return_code == 0 else "failed"
    return record


def checkpoint_path(model: str) -> Path:
    return RESULTS_DIR / f"best_{model}_model.pth"


def preserve_efficiency_artifacts(model: str, run_dir: Path) -> dict[Path, Path | None]:
    """Back up formal results so a one-epoch benchmark cannot overwrite them."""
    artifact_paths = [
        checkpoint_path(model),
        RESULTS_DIR / f"{model}_training_curve.png",
        RESULTS_DIR / "predictions" / model,
        RESULTS_DIR / "visualizations" / model,
    ]
    backup_root = run_dir / "preserved_results" / model
    state: dict[Path, Path | None] = {}
    for artifact in artifact_paths:
        if not artifact.exists():
            state[artifact] = None
            continue
        backup = backup_root / artifact.relative_to(RESULTS_DIR)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if artifact.is_dir():
            shutil.copytree(artifact, backup, dirs_exist_ok=True)
        else:
            shutil.copy2(artifact, backup)
        state[artifact] = backup
    return state


def restore_efficiency_artifacts(state: dict[Path, Path | None]) -> None:
    for artifact, backup in state.items():
        if artifact.is_dir():
            shutil.rmtree(artifact)
        elif artifact.exists():
            artifact.unlink()
        if backup is None:
            continue
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if backup.is_dir():
            shutil.copytree(backup, artifact)
        else:
            shutil.copy2(backup, artifact)


def write_runtime_csv(records: list[dict], path: Path) -> None:
    fields = [
        "model", "stage", "status", "return_code", "wall_seconds",
        "parameters_m", "epochs_completed", "train_seconds_mean", "train_seconds_total",
        "validation_seconds_mean", "inference_seconds",
        "mse", "mae", "mape", "r2", "command",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def validate_project(args: argparse.Namespace, models: list[str]) -> int:
    required = [
        MAIN_SCRIPT,
        PROJECT_ROOT / "train.py",
        PROJECT_ROOT / "evaluate.py",
        PROJECT_ROOT / "preprocessing.py",
        PROJECT_ROOT / "data" / "transformer_raw.csv",
        PROJECT_ROOT / "cluster" / "results" / "representative_transformers.txt",
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    print("Models:", ", ".join(models))
    print("Paper config:", PAPER_CONFIG)
    print("Python:", args.python)
    if missing:
        print("Missing required files:", ", ".join(missing))
        return 1
    result = subprocess.run(
        [args.python, "-m", "py_compile", "main.py", "train.py", "evaluate.py", "preprocessing.py"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    print("Validation passed." if result.returncode == 0 else "Python compilation failed.")
    return result.returncode


def main() -> int:
    args = parse_args()
    models = selected_models(args)
    if args.mode == "validate":
        return validate_project(args, models)

    if paper_settings_changed(args):
        print("WARNING: overridden training settings do not match the submitted manuscript.")

    run_name = args.run_name or datetime.now().strftime(f"%Y%m%d_%H%M%S_{args.mode}")
    run_dir = RESULTS_DIR / "experiment_logs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    successful_models: list[str] = []

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "models": models,
        "paper_config": PAPER_CONFIG,
        "effective_config": {
            "epochs": 1 if args.mode == "efficiency" else args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "python": args.python,
        "dry_run": args.dry_run,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.mode == "efficiency":
        for model in models:
            state = preserve_efficiency_artifacts(model, run_dir) if not args.dry_run else {}
            try:
                train_record = run_command(
                    command_for(args, "train", model=model, epochs=1),
                    run_dir / f"{model}_train.log", "train", model, args.dry_run,
                )
                records.append(train_record)
                if train_record["status"] in {"success", "dry-run"}:
                    predict_record = run_command(
                        command_for(args, "predict", model=model, epochs=1),
                        run_dir / f"{model}_predict.log", "predict", model, args.dry_run,
                    )
                    records.append(predict_record)
                elif args.fail_fast:
                    break
            finally:
                if not args.dry_run:
                    restore_efficiency_artifacts(state)
            write_runtime_csv(records, run_dir / "runtime_summary.csv")
    else:
        if args.mode in {"reproduce", "train"}:
            for model in models:
                if args.skip_existing and checkpoint_path(model).exists():
                    record = {"model": model, "stage": "train", "status": "skipped", "wall_seconds": 0.0}
                else:
                    record = run_command(
                        command_for(args, "train", model=model),
                        run_dir / f"{model}_train.log", "train", model, args.dry_run,
                    )
                records.append(record)
                if record["status"] in {"success", "skipped", "dry-run"}:
                    successful_models.append(model)
                elif args.fail_fast:
                    write_runtime_csv(records, run_dir / "runtime_summary.csv")
                    return 1
                write_runtime_csv(records, run_dir / "runtime_summary.csv")

        prediction_models = models if args.mode == "predict" else successful_models
        if args.mode in {"reproduce", "predict"}:
            for model in prediction_models:
                record = run_command(
                    command_for(args, "predict", model=model),
                    run_dir / f"{model}_predict.log", "predict", model, args.dry_run,
                )
                records.append(record)
                if record["status"] == "failed" and args.fail_fast:
                    write_runtime_csv(records, run_dir / "runtime_summary.csv")
                    return 1
                write_runtime_csv(records, run_dir / "runtime_summary.csv")

        if args.mode == "compare" or (args.mode == "reproduce" and len(successful_models) == len(models)):
            compare_record = run_command(
                command_for(args, "compare", models=models),
                run_dir / "compare.log", "compare", "all", args.dry_run,
            )
            records.append(compare_record)
        elif args.mode == "reproduce":
            print("Comparison skipped because at least one model failed to train.")

    write_runtime_csv(records, run_dir / "runtime_summary.csv")
    failed = [record for record in records if record.get("status") == "failed"]
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["status"] = "complete" if not failed else "incomplete"
    manifest["failed_commands"] = len(failed)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nExperiment records: {run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
