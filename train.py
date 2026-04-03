import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import evaluate
import preprocessing
import model
from model.hybrid_model import DualStreamPatchTST

# 尝试导入 TSLib 模型
try:
    from model.InformerModel import Model as Informer
    from model.Autoformer import Model as Autoformer
    from model.TimesNet import Model as TimesNet
    from model.PatchTST import Model as PatchTST
    from model.iTransformer import Model as iTransformer
    from model.Crossformer import Model as Crossformer
    from model.DARNN import Model as DARNN
except ImportError:
    print("TSLib models import failed. Check file paths.")

TSLIB_MODELS = ['Informer', 'Autoformer', 'TimesNet', 'PatchTST', 'iTransformer', 'Crossformer']


class Configs:
    """简单的配置包装类，适配 TSLib 接口"""
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            setattr(self, k, v)


def create_model(model_name, config):
    """根据名称创建模型实例"""

    # === 1. TSLib 系列模型 ===
    if model_name in TSLIB_MODELS:
        # TSLib 模型通用参数配置
        tslib_args_dict = {
            'task_name': 'long_term_forecast',
            'is_training': 1,
            'model_id': f'{model_name}_1',
            # 关键：输入输出维度匹配特征数量
            'enc_in': config['input_dim'],
            'dec_in': config['input_dim'],
            'c_out': config['input_dim'],
            'seq_len': config['seq_len'],
            'label_len': config['seq_len'] // 2,
            'pred_len': config['pred_len'],
            # 模型结构参数
            'd_model': 64 if model_name == 'DARNN' else 256,
            'n_heads': 4,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 1024,
            'dropout': config['dropout'],
            'attn': 'prob',
            'embed': 'timeF',
            'activation': 'gelu',
            'output_attention': False,
            'freq': 'h',

            # 其他特定参数
            'factor': 3,
            'moving_avg': 25,
            'distil': True,
            'top_k': 3,
            'num_kernels': 6,
            'patch_len': 16,
            'stride': 8,
            'seg_len': 6,
            'win_size': 2,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        }

        tslib_args = Configs(tslib_args_dict)

        if model_name == 'Informer':
            return Informer(tslib_args)
        elif model_name == 'Autoformer':
            return Autoformer(tslib_args)
        elif model_name == 'TimesNet':
            return TimesNet(tslib_args)
        elif model_name == 'PatchTST':
            print("Initializing PatchTST (Hybrid Mode: Load + Daily Temp)")
            # DailyGuidedPatchTST 需要 num_transformers 参数
            return DualStreamPatchTST(tslib_args, config.get('num_transformers'))
        elif model_name == 'iTransformer':
            return iTransformer(tslib_args)
        elif model_name == 'Crossformer':
            return Crossformer(tslib_args)
        elif model_name == 'DARNN':
            return DARNN(tslib_args)

    # === 2. 常规模型 ===
    model_classes = {
        'MLP': model.MLPModel.MLPModel,
        'CNN1D': model.CNN1DModel.CNN1DModel,
        'TCN': model.TCNModel.TCNModel,
        'RNN': model.RNNModel.RNNModel,
        'GRU': model.GRUModel.GRUModel,
        'LSTM': model.LSTMModel.LSTMModel,
        'CNN-LSTM': model.CNNLSTMModel.CNNLSTMModel,
        'Transformer-LSTM': model.TransformerLSTMModel.TransformerLSTMModel,
    }

    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")

    return model_classes[model_name](
        input_dim=config['input_dim'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        dropout=config['dropout']
    )


class CombinedLoss(nn.Module):
    """组合损失函数 (MSE + MAE)"""

    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, true):
        return self.alpha * self.mse(pred, true) + self.beta * self.mae(pred, true)


def save_model_safely(model, model_name, epoch, test_loss, metrics, config):
    """保存模型及配置"""
    os.makedirs('results', exist_ok=True)

    # 准备保存字典
    save_data = {
        'epoch': epoch,
        'test_loss': test_loss,
        'metrics': metrics,
        'config': config
    }

    if hasattr(model, 'module'):
        save_data['model_state_dict'] = model.module.state_dict()
    else:
        save_data['model_state_dict'] = model.state_dict()

    # 类型转换辅助函数
    def convert_for_saving(obj):
        if isinstance(obj, dict):
            return {k: convert_for_saving(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_saving(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)
        else:
            return obj

    save_data = convert_for_saving(save_data)
    model_path = f'results/best_{model_name}_model.pth'
    torch.save(save_data, model_path)
    print(f"Model saved: {model_path} (Loss: {test_loss:.6f})")


def train_epoch(model, dataloader, optimizer, criterion, device, model_name):
    """单轮训练逻辑"""
    model.train()
    total_loss = 0

    for batch_x, batch_y in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()
        batch_y = batch_y.to(device)

        # 处理不同模型的输入格式
        if model_name in TSLIB_MODELS:
            x_enc = batch_x['x_enc'].to(device)
            x_mark_enc = batch_x['x_mark_enc'].to(device)
            x_dec = batch_x['x_dec'].to(device)
            x_mark_dec = batch_x['x_mark_dec'].to(device)

            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if isinstance(outputs, tuple): outputs = outputs[0]

            # 截断输出以匹配标签
            if outputs.shape[1] > batch_y.shape[1]:
                outputs = outputs[:, -batch_y.shape[1]:, :]
        else:
            if isinstance(batch_x, dict):
                x_input = batch_x['x_enc'].to(device)
            else:
                x_input = batch_x.to(device)
            outputs = model(x_input)

        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(config, model_name, transformer_ids=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ==========================================
    # 数据加载分支
    # ==========================================
    if model_name == 'PatchTST':
        print("Using PatchTST Hybrid Data Loader (Load + Temp)...")

        # 1. 混合数据加载
        train_data, test_data, scaler, df_transformer, num_trans = preprocessing.load_hybrid_data_from_file(
            file_path=config['data_path'],
            train_ratio=config['train_ratio'],
            seq_len=config['seq_len'],
            pred_len=config['pred_len']
        )

        config['num_transformers'] = num_trans
        config['input_dim'] = num_trans

        dates = df_transformer.index
        train_size = len(df_transformer) * config['train_ratio']

        # 2. 混合数据集
        train_dataset = preprocessing.PatchTSTHybridDataset(
            train_data, dates[:int(train_size)], config['seq_len'], config['pred_len'])

        test_dataset = preprocessing.PatchTSTHybridDataset(
            test_data, dates[int(train_size) - config['seq_len']:], config['seq_len'], config['pred_len'])

    else:
        # 普通数据加载
        train_data, test_data, scaler, df_transformer = preprocessing.load_data_from_file(
            file_path=config['data_path'],
            train_ratio=config['train_ratio'],
            seq_len=config['seq_len'],
            pred_len=config['pred_len']
        )

        config['input_dim'] = train_data.shape[1]

        # 数据集构建
        if model_name in TSLIB_MODELS:
            dates = df_transformer.index
            train_size = len(train_data)
            train_dataset = preprocessing.Dataset_TSLib(
                train_data, dates[:train_size], config['seq_len'], config['pred_len'])
            test_dataset = preprocessing.Dataset_TSLib(
                test_data, dates[train_size - config['seq_len']:], config['seq_len'], config['pred_len'])
        else:
            train_dataset = preprocessing.TransformerDataset(train_data, config['seq_len'], config['pred_len'])
            test_dataset = preprocessing.TransformerDataset(test_data, config['seq_len'], config['pred_len'])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=False)

    # 模型初始化
    model = create_model(model_name, config).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = CombinedLoss()

    # 训练循环
    best_test_loss = float('inf')
    best_metrics = None
    train_losses = []
    test_losses = []

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, model_name)
        train_losses.append(train_loss)

        test_metrics, _, _ = evaluate.evaluate(model, test_loader, criterion, device, model_name)
        test_loss = test_metrics['overall']['loss']
        test_losses.append(test_loss)

        print(f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        print(f"MSE: {test_metrics['overall']['mse']:.6f}, R2: {test_metrics['overall']['r2']:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_metrics = test_metrics
            save_model_safely(model, model_name, epoch, test_loss, test_metrics, config)

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Curve')
    plt.legend()
    plt.grid(True)

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    return model, scaler, best_metrics