import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# 尝试导入时间特征工具
try:
    from utils.timefeatures import time_features
except ImportError:
    print("Warning: utils.timefeatures not found.")


class TransformerDataset(Dataset):
    """
    通用模型数据集 (LSTM, MLP, CNN等)
    支持自定义步长 (Stride) 实现稀疏采样
    """

    def __init__(self, data, seq_len, pred_len, stride=1):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride

    def __len__(self):
        target_len = len(self.data) - self.seq_len - self.pred_len
        if target_len < 0:
            return 0
        return target_len // self.stride + 1

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len
        pred_end_idx = end_idx + self.pred_len

        seq = self.data[start_idx:end_idx]
        label = self.data[end_idx:pred_end_idx]

        return torch.FloatTensor(seq), torch.FloatTensor(label)


class Dataset_TSLib(Dataset):
    """
    TSLib 系列模型专用数据集 (Informer, Autoformer, iTransformer等)
    特点: 同时返回 x_enc, x_dec, x_mark_enc, x_mark_dec
    """

    def __init__(self, data, dates, seq_len, pred_len, label_len=None, freq='h', stride=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len if label_len is not None else int(seq_len / 2)
        self.stride = stride

        self.data_x = data.astype(np.float32)
        self.data_y = data.astype(np.float32)

        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)

        df_stamp = pd.DataFrame({'date': dates})
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)

        self.data_stamp = data_stamp.astype(np.float32)

    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 1. Encoder Input
        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        # 2. Decoder Input (Start Token + Zero Padding)
        dec_inp_token = self.data_x[r_begin:r_begin + self.label_len]
        dec_inp_zeros = np.zeros((self.pred_len, self.data_x.shape[-1]), dtype=np.float32)
        seq_x_dec = np.concatenate([dec_inp_token, dec_inp_zeros], axis=0)

        # 3. Decoder Time Features
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # 4. Target
        seq_y = self.data_y[s_end:s_end + self.pred_len]

        return {
            'x_enc': torch.tensor(seq_x),
            'x_mark_enc': torch.tensor(seq_x_mark),
            'x_dec': torch.tensor(seq_x_dec),
            'x_mark_dec': torch.tensor(seq_y_mark)
        }, torch.tensor(seq_y)

    def __len__(self):
        return max(0, (len(self.data_x) - self.seq_len - self.pred_len) // self.stride + 1)


def load_data_from_file(file_path, train_ratio=0.8, seq_len=96, pred_len=24):
    """常规数据加载：直接读取 CSV 并标准化"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    print(f"Loading dataset: {file_path}")
    df = pd.read_csv(file_path)

    if 'DATETIME' in df.columns:
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df.set_index('DATETIME', inplace=True)
    else:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)

    data_values = df.values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)

    train_size = int(len(data_scaled) * train_ratio)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size - seq_len:]

    print(f"Data shape: {data_scaled.shape}")
    return train_data, test_data, scaler, df


def generate_processed_file(raw_path, target_ids, output_path):
    """ETL: 筛选代表性变压器 -> 宽表转换 -> 缺失值填充"""
    if os.path.exists(output_path):
        print(f"Processed file exists: {output_path}")
        return

    print(f"Processing raw data: {raw_path}")

    if raw_path.endswith('.csv'):
        df = pd.read_csv(raw_path)
    else:
        df = pd.read_excel(raw_path)

    if target_ids is not None:
        df = df[df['TRANSFORMER_ID'].isin(target_ids)].copy()

    if len(df) == 0:
        raise ValueError("筛选后数据为空")

    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    # Long to Wide
    df_pivot = df.pivot_table(index='DATETIME', columns='TRANSFORMER_ID', values='LOAD', aggfunc='mean')

    # Resample & Interpolate
    df_pivot = df_pivot.resample('1h').asfreq()
    df_pivot = df_pivot.interpolate(method='linear').bfill().ffill()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_pivot.to_csv(output_path)
    print(f"Saved processed data to: {output_path}")


def load_hybrid_data_from_file(file_path, train_ratio=0.8, seq_len=96, pred_len=24):
    """
    [PatchTST 专用] 加载混合数据
    返回: (Load_Data, Temp_Data) 元组
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"混合数据文件不存在: {file_path}")

    print(f"Loading hybrid dataset: {file_path}")
    df = pd.read_csv(file_path)

    # 索引处理
    if 'DATETIME' in df.columns:
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df.set_index('DATETIME', inplace=True)
    else:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)

    # 分离 Load 和 Temp 列
    cols = df.columns
    temp_cols = [c for c in cols if c.startswith('TEMP_')]
    load_cols = [c for c in cols if not c.startswith('TEMP_')]
    load_cols.sort()

    # 确保每个 Load 都有对应的 Temp (没有则补0)
    expected_temp_cols = [f"TEMP_{c}" for c in load_cols]
    final_temp_cols = []

    for tc in expected_temp_cols:
        if tc in df.columns:
            final_temp_cols.append(tc)
        else:
            df[tc] = 0
            final_temp_cols.append(tc)

    df_load = df[load_cols]
    df_temp = df[final_temp_cols]

    # 分别归一化
    scaler_load = StandardScaler()
    scaler_temp = StandardScaler()

    data_load = scaler_load.fit_transform(df_load.values)
    data_temp = scaler_temp.fit_transform(df_temp.values)

    train_size = int(len(data_load) * train_ratio)

    # 构造返回数据
    train_data = (data_load[:train_size], data_temp[:train_size])
    test_data = (data_load[train_size - seq_len:], data_temp[train_size - seq_len:])

    print(f"Hybrid Load Features: {data_load.shape[1]}, Temp Features: {data_temp.shape[1]}")
    return train_data, test_data, scaler_load, df, len(load_cols)


class PatchTSTHybridDataset(Dataset):
    """
    [PatchTST 专用] 混合数据集
    Input: [Load, Temp] (Concatenated)
    Output: [Load] (Only)
    """

    def __init__(self, data_pack, dates, seq_len, pred_len, label_len=None, freq='h', stride=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len if label_len is not None else int(seq_len / 2)
        self.stride = stride

        data_load, data_temp = data_pack
        self.num_transformers = data_load.shape[1]

        # Input: Load + Temp
        self.data_x = np.concatenate([data_load, data_temp], axis=1).astype(np.float32)
        # Target: Load only
        self.data_y = data_load.astype(np.float32)

        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)

        df_stamp = pd.DataFrame({'date': dates})
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp.astype(np.float32)

    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Encoder Input (Load + Temp)
        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        # Decoder Input (Load only + Zeros)
        dec_inp_token = self.data_y[r_begin:r_begin + self.label_len]
        dec_inp_zeros = np.zeros((self.pred_len, self.num_transformers), dtype=np.float32)

        seq_x_dec = np.concatenate([dec_inp_token, dec_inp_zeros], axis=0)
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # Target (Load only)
        seq_y = self.data_y[s_end:s_end + self.pred_len]

        return {
            'x_enc': torch.tensor(seq_x),
            'x_mark_enc': torch.tensor(seq_x_mark),
            'x_dec': torch.tensor(seq_x_dec),
            'x_mark_dec': torch.tensor(seq_y_mark)
        }, torch.tensor(seq_y)

    def __len__(self):
        return max(0, (len(self.data_x) - self.seq_len - self.pred_len) // self.stride + 1)