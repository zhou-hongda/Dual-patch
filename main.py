import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import warnings

# PyTorch emits this warning for the pre-norm Transformer configuration used here.
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

import train
import evaluate
import preprocessing
from merge_data import merge_weather_correctly


class DataPaths:
    """Project data paths, resolved relative to the project root."""

    RAW = 'data/transformer_raw.csv'
    PROCESSED = 'data/representative_data.csv'
    WEATHER = 'data/representative_data_with_weather.csv'
    CLUSTER_IDS = 'cluster/results/representative_transformers.txt'


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_representative_ids(filepath):
    """加载 ID 列表"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"缺少必要文件: {filepath}")

    with open(filepath, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    print(f"已加载 {len(ids)} 个 ID")
    return ids


def main():
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTM')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'compare'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--compare_models', type=str, nargs='+')
    args = parser.parse_args()

    base_config = {
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'seq_len': 168,
        'pred_len': 24,
        'input_dim': 1,
        'dropout': 0.1,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4
    }

    representative_ids = load_representative_ids(DataPaths.CLUSTER_IDS)

    preprocessing.generate_processed_file(
        raw_path=DataPaths.RAW,
        target_ids=representative_ids,
        output_path=DataPaths.PROCESSED
    )

    is_patchtst = (args.model == 'PatchTST') or \
                  (args.compare_models and 'PatchTST' in args.compare_models)

    current_num_transformers = len(representative_ids)

    if is_patchtst:
        print("准备 Hybrid PatchTST 数据环境...")
        merge_weather_correctly()
        base_config['data_path'] = DataPaths.WEATHER

        try:
            df_head = pd.read_csv(DataPaths.WEATHER, nrows=0)
            load_cols = [c for c in df_head.columns if c != 'DATETIME' and not c.startswith('TEMP_')]
            current_num_transformers = len(load_cols)
            print(f"检测到变压器数量: {current_num_transformers}")
        except Exception:
            print("列数检测失败，使用 ID 列表长度作为默认值")
    else:
        base_config['data_path'] = DataPaths.PROCESSED

    base_config['num_transformers'] = current_num_transformers

    if args.mode == 'train':
        print(f"开始训练 {args.model} ...")
        train.train_model(base_config, args.model, transformer_ids=None)

    elif args.mode == 'predict':
        print(f"开始预测: {args.model}")
        test_stride = 1
        evaluate.predict_mode(args.model, base_config, transformer_ids=None, stride=test_stride)

    elif args.mode == 'compare':
        if not args.compare_models:
            print("请提供对比模型: --compare_models ModelA ModelB")
            return

        print(f"开始对比: {args.compare_models}")
        evaluate.compare_models(
            args.compare_models,
            base_config,
            transformer_ids=None,
            stride=1
        )


if __name__ == "__main__":
    main()
