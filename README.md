# Dual-Patch

Reproducibility code for **A Physics-Guided Dual-Stream Independent Transformer
for Regional Residential Building Load Forecasting**.

Dual-Patch processes transformer load and matched ambient temperature in separate
PatchTST-style encoders. A temperature-driven gate then guides their fusion before
multi-step load prediction.

## Repository layout

- `main.py`: training and prediction command-line interface.
- `model/hybrid_model.py`: Dual-Patch architecture.
- `preprocessing.py`: chronological split and train-only normalization.
- `train.py`: optimization and best-validation checkpointing.
- `evaluate.py`: checkpoint evaluation on the held-out test split.
- `run_experiments.py`: one-command validation and reproduction.
- `scripts/clustering.py`: representative-transformer selection utility.
- `config/representative_transformers.txt`: representative IDs used in the paper.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The raw load dataset is not stored in Git because it exceeds GitHub's file-size
limit. Place the prepared hybrid data at
`data/representative_data_with_weather.csv`, or pass another path with
`--data-path`.

The CSV must contain a `DATETIME` column and paired columns for every transformer:

```text
DATETIME,5-2-7,2-0-5,TEMP_5-2-7,TEMP_2-0-5
2021-01-01 00:00:00,...
```

Load columns are sorted by name and each must have a corresponding `TEMP_<ID>`
column. Splits are chronological (70% train, 15% validation, 15% test), and both
scalers are fitted on the training period only.

## Usage

Validate the code and report whether the external dataset is available:

```bash
python run_experiments.py --mode validate
```

Train the paper model and save the best validation checkpoint:

```bash
python main.py --mode train --epochs 15 --batch-size 64
```

Evaluate that checkpoint on the test split:

```bash
python main.py --mode predict
```

Run training followed by evaluation:

```bash
python run_experiments.py --mode reproduce
```

Checkpoints and JSON metric files are written under `checkpoints/`. Experiment-run
metadata is written under `results/experiment_logs/`; generated outputs are ignored
by Git.

## Preservation

The original submitted code is preserved by the local Git tag
`original-code-backup`. The release workflow does not modify that tagged commit.
