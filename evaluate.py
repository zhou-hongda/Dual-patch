import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import preprocessing
from tqdm import tqdm

# --- 绘图配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 模型列表
TSLIB_MODELS = ['iTransformer', 'Crossformer', 'PatchTST', 'Informer', 'Reformer', 'Autoformer', 'TimesNet', 'DARNN']


def robust_inverse_transform(scaler, data_tensor, n_samples, n_timesteps, n_features):
    """
    处理 Hybrid 模型反归一化时的维度不匹配问题。
    如果 Scaler 期待 8 列 (Load+Temp)，但 Data 只有 4 列 (Load)，
    则补 0 后还原，再截取前 4 列。
    """
    # 1. 展平数据
    data_flat = data_tensor.reshape(-1, n_features)

    # 2. 获取 Scaler 期望的特征数
    scaler_features = scaler.mean_.shape[0]

    # 3. 判断是否需要填充
    if scaler_features > n_features:
        dummy = np.zeros((data_flat.shape[0], scaler_features))
        dummy[:, :n_features] = data_flat
        inversed_dummy = scaler.inverse_transform(dummy)
        inversed_data = inversed_dummy[:, :n_features]
    else:
        inversed_data = scaler.inverse_transform(data_flat)

    # 4. 还原形状
    return inversed_data.reshape(n_samples, n_timesteps, n_features)


# =========================================================
# 1. 核心评估函数
# =========================================================
def evaluate(model, dataloader, criterion, device, model_name='LSTM'):
    """
    计算验证集/测试集的 Loss 和基础指标。
    """
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_y = batch_y.to(device)

            # --- 分支 1: TSLib 模型 ---
            if model_name in TSLIB_MODELS:
                x_enc = batch_x['x_enc'].to(device)
                x_mark_enc = batch_x['x_mark_enc'].to(device)
                x_dec = batch_x['x_dec'].to(device)
                x_mark_dec = batch_x['x_mark_dec'].to(device)

                outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                if isinstance(outputs, tuple): outputs = outputs[0]
                outputs = outputs.to(device)

                if outputs.shape[1] > batch_y.shape[1]:
                    outputs = outputs[:, -batch_y.shape[1]:, :]

            # --- 分支 2: 普通模型 ---
            else:
                if isinstance(batch_x, dict):
                    input_tensor = batch_x['x_enc'].to(device)
                else:
                    input_tensor = batch_x.to(device)

                outputs = model(input_tensor)

            # 维度修正
            if outputs.shape != batch_y.shape:
                if outputs.shape[-1] == 1 and batch_y.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # 计算指标
    metrics_values = evaluate_metrics(predictions, actuals)
    metrics_values['loss'] = total_loss / len(dataloader)
    metrics = {'overall': metrics_values}

    return metrics, predictions, actuals


def evaluate_metrics(predictions, actuals):
    """通用指标计算"""
    flat_pred = predictions.reshape(-1)
    flat_act = actuals.reshape(-1)

    mse = mean_squared_error(flat_act, flat_pred)
    mae = np.mean(np.abs(flat_pred - flat_act))
    mape = mean_absolute_percentage_error(flat_act, flat_pred) * 100
    r2 = r2_score(flat_act, flat_pred)

    return {'mse': mse, 'mae': mae, 'mape': mape, 'r2': r2}


def evaluate_step_metrics(preds_norm, acts_norm, preds_orig, acts_orig, steps=[0, 7, 15, 23]):
    """计算特定预测步长（兼顾归一化MSE与真实量纲）的评价指标"""
    step_metrics = {}
    for step in steps:
        if step >= preds_norm.shape[1]:
            continue

        # 1. 归一化数据计算 MSE (Normalized)
        pred_n = preds_norm[:, step, :].reshape(-1)
        act_n = acts_norm[:, step, :].reshape(-1)
        mse = mean_squared_error(act_n, pred_n)

        # 2. 原始反归一化数据计算 MAE, MAPE, R2 (真实物理量纲)
        pred_o = preds_orig[:, step, :].reshape(-1)
        act_o = acts_orig[:, step, :].reshape(-1)
        mae = np.mean(np.abs(pred_o - act_o))
        mape = mean_absolute_percentage_error(act_o, pred_o) * 100
        r2 = r2_score(act_o, pred_o)

        step_metrics[step] = {'mse': mse, 'mae': mae, 'mape': mape, 'r2': r2}

    return step_metrics
def generate_evaluation_report(metrics, config, model_name):
    """生成评估报告"""
    print(f"\n--- {model_name} Evaluation Report ---")
    print(f"Config: Seq_Len={config['seq_len']}, Pred_Len={config['pred_len']}")
    print(f"Test MSE:  {metrics['overall']['mse']:.5f}")
    print(f"Test MAPE: {metrics['overall']['mape']:.2f}%")
    print(f"Test R2:   {metrics['overall']['r2']:.5f}")
    print("----------------------------------\n")


# =========================================================
# 2. 辅助功能函数 (加载与预测)
# =========================================================
def load_trained_model(model_name, config, device):
    import train
    model_path = f'results/best_{model_name}_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    saved_config = checkpoint.get('config', {})
    if 'input_dim' in saved_config: config['input_dim'] = saved_config['input_dim']

    model = train.create_model(model_name, config)
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model


def make_predictions(model, test_data, seq_len, pred_len, device, model_name, df_transformer=None, stride=None):
    """
    生成预测结果 (支持动态计算变压器数量)
    """
    model.eval()
    predictions, actuals = [], []

    if stride is None:
        stride = 1

    test_start_idx = 0
    if df_transformer is not None:
        test_start_idx = len(df_transformer) - len(test_data)

    label_len = seq_len // 2
    total_windows = (len(test_data) - seq_len - pred_len) // stride + 1
    print(f"{model_name}: Generating predictions (Windows: {total_windows}, Stride: {stride})...")

    # =========================================================
    # 动态计算变压器数量 (N_TRANSFORMERS)
    # =========================================================
    if df_transformer is not None:
        load_cols = [c for c in df_transformer.columns if not c.startswith('TEMP_')]
        n_real_transformers = len(load_cols)
    else:
        n_real_transformers = test_data.shape[1]

    with torch.no_grad():
        loop_range = range(0, len(test_data) - seq_len - pred_len + 1, stride)

        for i in tqdm(loop_range, desc="Inference", unit="window"):
            seq_x = test_data[i:i + seq_len]
            seq_y = test_data[i + seq_len:i + seq_len + pred_len]

            # === 分支 A: TSLib 系列模型 ===
            if model_name in TSLIB_MODELS:
                s_end = i + seq_len
                r_begin = s_end - label_len
                r_end = r_begin + label_len + pred_len

                abs_s_begin = test_start_idx + i
                abs_s_end = test_start_idx + s_end
                abs_r_begin = test_start_idx + r_begin
                abs_r_end = test_start_idx + r_end

                # 获取时间特征
                date_enc = df_transformer.index[abs_s_begin: abs_s_end]
                date_dec = df_transformer.index[abs_r_begin: abs_r_end]

                try:
                    from utils.timefeatures import time_features
                    data_stamp_enc = time_features(pd.to_datetime(date_enc.values), freq='h')
                    data_stamp_dec = time_features(pd.to_datetime(date_dec.values), freq='h')
                    if data_stamp_enc.shape[0] == 4: data_stamp_enc = data_stamp_enc.T
                    if data_stamp_dec.shape[0] == 4: data_stamp_dec = data_stamp_dec.T
                except ImportError:
                    data_stamp_enc = np.zeros((len(date_enc), 4))
                    data_stamp_dec = np.zeros((len(date_dec), 4))

                x_enc = torch.FloatTensor(seq_x).unsqueeze(0).to(device)
                x_mark_enc = torch.FloatTensor(data_stamp_enc).unsqueeze(0).to(device)

                # Decoder Input
                dec_inp_token = seq_x[-label_len:]
                dec_inp_zeros = np.zeros((pred_len, seq_x.shape[-1]))
                x_dec = torch.FloatTensor(np.concatenate([dec_inp_token, dec_inp_zeros], axis=0)).unsqueeze(0).to(
                    device)
                x_mark_dec = torch.FloatTensor(data_stamp_dec).unsqueeze(0).to(device)

                # 动态修复 1: Crossformer 输入截断
                if model_name != 'PatchTST' and x_enc.shape[-1] > n_real_transformers:
                    x_enc = x_enc[:, :, :n_real_transformers]
                    x_dec = x_dec[:, :, :n_real_transformers]

                pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # === 分支 B: 普通模型 ===
            else:
                x_enc = torch.FloatTensor(seq_x).unsqueeze(0).to(device)

                # 普通模型输入截断
                if model_name != 'PatchTST' and x_enc.shape[-1] > n_real_transformers:
                    x_enc = x_enc[:, :, :n_real_transformers]

                pred = model(x_enc)

            if isinstance(pred, tuple): pred = pred[0]
            if pred.shape[1] > pred_len: pred = pred[:, -pred_len:, :]

            # 动态修复 2: 统一输出截断
            if pred.shape[-1] > n_real_transformers:
                pred = pred[:, :, :n_real_transformers]

            predictions.append(pred.cpu().numpy()[0])

            # 动态修复 3: 真实值截断
            if seq_y.shape[1] > n_real_transformers:
                seq_y = seq_y[:, :n_real_transformers]

            actuals.append(seq_y)

    return np.array(predictions), np.array(actuals)


# =========================================================
# 3. 绘图功能：步长分层图
# =========================================================
def plot_horizon_lines(predictions, actuals, model_name, feature_names):
    save_dir = f'results/visualizations/{model_name}'
    os.makedirs(save_dir, exist_ok=True)

    pred_len = predictions.shape[1]
    num_features = predictions.shape[2]
    display_limit = 300

    print(f"Generating horizon lines for {model_name}...")
    cmap = plt.cm.jet
    colors = cmap(np.linspace(0, 1, pred_len))

    for f_idx in range(num_features):
        feature_name = feature_names[f_idx]
        plt.figure(figsize=(20, 8))

        gt_len = min(len(actuals), display_limit + pred_len + 5)
        gt_series = actuals[:gt_len, 0, f_idx]
        plt.plot(range(len(gt_series)), gt_series, color='black', linewidth=2.5, label='Actual', zorder=10)

        for step in range(pred_len):
            step_series = predictions[:display_limit, step, f_idx]
            x_range = range(step, step + len(step_series))
            plt.plot(x_range, step_series, color=colors[step], linewidth=1, alpha=0.6)

        plt.title(f'{model_name} | {feature_name} | Horizon Plot (Step 1-{pred_len})', fontsize=16)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Load', fontsize=12)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=pred_len))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01)
        cbar.set_label('Prediction Step (1=Blue ... 24=Red)', rotation=270, labelpad=15)

        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        safe_name = str(feature_name).replace('/', '_')
        plt.savefig(f'{save_dir}/{safe_name}_horizon_24lines.png', bbox_inches='tight')
        plt.close()


# =========================================================
# 4. 对比绘图功能
# =========================================================
def plot_multi_model_step_comparison(all_preds, actuals, model_names, feature_names):
    """
    多模型预测步长对比折线图 - 论文专用版 (600 DPI 高清版)
    1. 字体全局替换为 Times New Roman
    2. 布局改为 1x4 横向排列
    3. 全局标号：每张变压器的大图分别标记为 (a), (b), (c), (d)...
    4. 调细真实值黑线的粗细
    5. DPI 提升至 600 满足顶级期刊印刷要求
    """
    save_dir = 'results/visualizations/comparison'
    import os
    os.makedirs(save_dir, exist_ok=True)

    # === 设置字体为 Times New Roman ===
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    steps_to_plot = [0, 7, 15, 23]
    step_labels = [1, 8, 16, 24]
    max_len = list(all_preds.values())[0].shape[1]
    valid_indices = [i for i, s in enumerate(steps_to_plot) if s < max_len]
    steps_to_plot = [steps_to_plot[i] for i in valid_indices]
    step_labels = [step_labels[i] for i in valid_indices]
    num_features = actuals.shape[2]

    display_limit = 200

    print("Generating multi-model step comparison plot (1x4 Layout, overall figure labels, 600 DPI)...")

    for f_idx in range(num_features):
        feature_name = feature_names[f_idx]

        # === 核心修改 1: 根据循环索引动态生成整图的 (a), (b), (c), (d) 标号 ===
        fig_label = f"({chr(97 + f_idx)})"

        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        axes = axes.flatten()
        gt_base = actuals[:display_limit, :, f_idx]

        for i, step in enumerate(steps_to_plot):
            if i >= len(axes): break
            ax = axes[i]
            gt = gt_base[:, step]

            # 黑线调细
            ax.plot(gt, color='black', linewidth=1.5, label='Actual', zorder=10)

            styles = ['--', '-.', ':']
            for m_idx, name in enumerate(model_names):
                if name not in all_preds: continue
                pred = all_preds[name][:display_limit, step, f_idx]
                style = styles[m_idx % len(styles)]
                ax.plot(pred, linestyle=style, linewidth=1.5, label=name, alpha=0.8)

            # === 核心修改 2: 移除子图里的标号，只保留 Step 名字 ===
            ax.set_title(f'Step {step_labels[i]} (+{step_labels[i]}h)',
                         fontsize=14, fontweight='bold', pad=10)

            ax.set_xlabel('Time Step', fontsize=12)

            if i == 0:
                ax.set_ylabel('Load', fontsize=12)
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

            ax.grid(True, alpha=0.3, linestyle='--')

        # === 核心修改 3: 将 (a)(b)(c) 加到整张图片的全局大标题中 ===
        plt.suptitle(f'{fig_label} Multi-Model Comparison | Feature: {feature_name}',
                     fontsize=18, fontweight='bold', y=1.05)

        plt.tight_layout()
        safe_name = str(feature_name).replace('/', '_')

        # 保存为高清无损 PNG 格式，DPI=600
        plt.savefig(f'{save_dir}/{safe_name}_compare_steps_1x4.png', dpi=600, bbox_inches='tight')
        plt.close()

    plt.rcParams['font.family'] = 'sans-serif'
def plot_metrics_comparison_combined(all_metrics):
    """
    指标对比 (4合1大图) - 论文专用版 (视觉优化版)
    1. 增加画布宽度，避免柱子过挤
    2. 缩小柱子上的数值字号并加粗
    3. 增大 X 轴标签倾斜角并加粗，防止长模型名重叠
    """
    save_dir = 'results/visualizations/comparison'
    os.makedirs(save_dir, exist_ok=True)

    # === 设置字体为 Times New Roman ===
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    metrics_names = ['mse', 'mae', 'mape', 'r2']
    titles = ['(a) MSE (Normalized)', '(b) MAE', '(c) MAPE', '(d) R²']
    models = list(all_metrics.keys())
    unified_color = '#4c72b0'

    # 优化 1：将宽度从 16 增加到 18，让 11 个模型有更充足的横向空间
    print("Generating combined metrics comparison plot (Optimized visual)...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for i, met in enumerate(metrics_names):
        ax = axes[i]
        original_values = [all_metrics[m][met] for m in models]

        # --- 核心修复：移除除以 max_val 的强制缩放，直接画真实值 ---
        plot_values = original_values

        if met == 'mse':
            y_label = 'MSE (Normalized)'
        elif met == 'r2':
            y_label = 'R²'
        else:
            y_label = met.upper()

        bars = ax.bar(models, plot_values, color=unified_color, alpha=0.85, width=0.6, edgecolor='black')

        ax.set_title(titles[i], fontsize=18, fontweight='bold', pad=15)
        ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        v_min, v_max = min(plot_values), max(plot_values)
        v_range = v_max - v_min

        if v_range == 0:
            bottom_limit = v_min * 0.9 if v_min > 0 else v_min * 1.1
            top_limit = v_max * 1.1 if v_max > 0 else v_max * 0.9
        else:
            margin = v_range * 0.25
            bottom_limit = v_min - margin
            if met == 'r2' and bottom_limit < 0 and v_min >= 0:
                bottom_limit = 0
            top_limit = v_max + margin * 1.5

        ax.set_ylim(bottom=bottom_limit, top=top_limit)

        for bar in bars:
            height = bar.get_height()
            text_y_offset = height + (v_range * 0.03 if v_range > 0 else height * 0.02)

            # 统一所有指标为 4 位小数，去掉了 MAPE 的百分号以节省空间
            text_val = f'{height:.4f}'

            # 优化 2：将柱子顶部的数值字号缩小到 10，并加粗 (fontweight='bold')
            ax.text(bar.get_x() + bar.get_width() / 2., text_y_offset, text_val,
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 优化 3：X 轴标签字号设为 12，加粗，并将旋转角度从 15度 增加到 30度
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=12, fontweight='bold')
        plt.setp(ax.get_yticklabels(), fontsize=12)

    plt.tight_layout(pad=3.0)
    plt.savefig(f'{save_dir}/comparison_all_metrics.png', dpi=600, bbox_inches='tight')
    plt.close()

    plt.rcParams['font.family'] = 'sans-serif'
# =========================================================
# 5. 主流程
# =========================================================
def compare_models(model_names, config, transformer_ids=None, stride=1):
    """
    对比模式入口 (支持稀疏窗口 & Chronos)
    """
    print(f"Starting comparison: {model_names} (Stride={stride})")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载数据
    _, test_data, scaler, df_transformer = preprocessing.load_data_from_file(
        file_path=config['data_path'],
        seq_len=config['seq_len'], pred_len=config['pred_len']
    )
    feature_names = df_transformer.columns.tolist()
    config['input_dim'] = test_data.shape[1]

    all_preds_orig = {}
    all_metrics = {}
    all_step_metrics = {}  # 新增：用于存储所有模型的分步指标
    actuals_orig = None


    for name in model_names:
        try:
            # === 模型加载逻辑 ===
            if name == 'Chronos':
                print(f"Loading Chronos (Adapter)...")
                from chronos_adapter import ChronosAdapter

                finetuned_path = "results/finetuned_Chronos"
                if os.path.exists(finetuned_path):
                    print(f"Loading finetuned weights: {finetuned_path}")
                    model = ChronosAdapter(config, model_id=finetuned_path).to(device)
                else:
                    print("Using pretrained weights")
                    model = ChronosAdapter(config).to(device)
            else:
                model = load_trained_model(name, config, device)

            # === 预测逻辑 ===
            preds, acts = make_predictions(
                model, test_data, config['seq_len'], config['pred_len'],
                device, name, df_transformer, stride=stride
            )

            n_samples, n_timesteps, n_features = preds.shape

            # 双重维度检查
            if acts.shape[-1] > n_features:
                acts = acts[:, :, :n_features]

            print(f"{name}: Performing robust inverse transform...")

            pred_orig = robust_inverse_transform(scaler, preds, n_samples, n_timesteps, n_features)
            act_orig = robust_inverse_transform(scaler, acts, n_samples, n_timesteps, n_features)

            if actuals_orig is None: actuals_orig = act_orig

            all_preds_orig[name] = pred_orig

            # --- 核心修改：分离指标计算 ---
            # 1. 先计算基于反归一化（真实量纲）的 MAE, MAPE, R2
            metrics_orig = evaluate_metrics(pred_orig, act_orig)

            # 2. 计算基于归一化数据的 MSE，并覆盖原字典中的 mse
            norm_mse = mean_squared_error(acts.reshape(-1), preds.reshape(-1))
            metrics_orig['mse'] = norm_mse

            all_metrics[name] = metrics_orig
            print(f"{name} Normalized MSE: {all_metrics[name]['mse']:.4f}")
            # --- 新增：计算特定步长(1, 8, 16, 24)的分步指标 ---
            steps_to_eval = [0, 7, 15, 23]
            all_step_metrics[name] = evaluate_step_metrics(
                preds, acts, pred_orig, act_orig, steps=steps_to_eval
            )
        except Exception as e:
            print(f"Model {name} failed: {e}")
            import traceback
            traceback.print_exc()

    # === 结果汇总与导出 ===
    if len(all_preds_orig) > 0:
        plot_multi_model_step_comparison(all_preds_orig, actuals_orig, list(all_preds_orig.keys()), feature_names)
        plot_metrics_comparison_combined(all_metrics)

        print("Exporting comparison data (Step-1)...")
        save_dir = 'results/visualizations/comparison'
        os.makedirs(save_dir, exist_ok=True)

        # 修复时间索引
        test_start_idx = len(df_transformer) - len(test_data)
        first_pred_time_idx = test_start_idx + config['seq_len']

        n_samples = actuals_orig.shape[0]
        full_index = df_transformer.index
        time_index = full_index[first_pred_time_idx:: stride]

        if len(time_index) > n_samples:
            time_index = time_index[:n_samples]

        if len(time_index) != n_samples:
            print(f"Warning: Index length mismatch ({len(time_index)} vs {n_samples}). Force aligning.")
            min_len = min(len(time_index), n_samples)
            time_index = time_index[:min_len]
            actuals_orig = actuals_orig[:min_len]
            for k in all_preds_orig:
                all_preds_orig[k] = all_preds_orig[k][:min_len]

        target_feature_idx = 0
        target_feature_name = feature_names[target_feature_idx]

        df_compare = pd.DataFrame(index=time_index)
        df_compare['Time'] = time_index
        df_compare[f'Actual_{target_feature_name}'] = actuals_orig[:, 0, target_feature_idx]

        for name, pred_data in all_preds_orig.items():
            df_compare[f'{name}_Pred'] = pred_data[:, 0, target_feature_idx]

        csv_path = f'{save_dir}/all_models_comparison_step1.csv'
        df_compare.to_csv(csv_path, index=False)
        print(f"Comparison data saved to: {csv_path}")
        # --- 新增：导出分步指标纵向对比大表格 (顶刊格式) ---
        print("Exporting multi-step metrics table (Vertical formatting)...")

        steps_to_eval = [0, 7, 15, 23]  # 对应真实的 Step 1, 8, 16, 24
        metrics_keys = ['mse', 'r2', 'mae', 'mape']
        metrics_display = ['MSE', 'R²', 'MAE', 'MAPE']

        rows = []
        for step_idx in steps_to_eval:
            step_label = f"Multi-step {step_idx + 1}"

            for m_idx, metric_key in enumerate(metrics_keys):
                row = {}
                # 模仿顶刊排版：只在每个步长的第一行显示步长名称，其余留空
                row['Multi-step'] = step_label if m_idx == 0 else ""
                row['Model evaluation indicators'] = metrics_display[m_idx]

                # 遍历所有模型，将其作为列
                for model_name in all_metrics.keys():  # 确保按照传入模型的顺序
                    if model_name in all_step_metrics:
                        val = all_step_metrics[model_name][step_idx][metric_key]
                        # 统一保留4位小数以对齐表格
                        row[model_name] = f"{val:.4f}"

                rows.append(row)

        df_step_metrics = pd.DataFrame(rows)
        csv_step_path = f'{save_dir}/multi_step_metrics_table_vertical.csv'
        df_step_metrics.to_csv(csv_step_path, index=False)
        print(f"Vertical multi-step metrics table saved to: {csv_step_path}")

def predict_mode(model_name, config, transformer_ids=None, stride=1):
    """
    独立预测模式 (支持稀疏窗口)
    """
    print(f"Starting prediction: {model_name} (Stride={stride})")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载数据
    _, test_data, scaler, df_transformer = preprocessing.load_data_from_file(
        file_path=config['data_path'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len']
    )

    feature_names = df_transformer.columns.tolist()
    config['input_dim'] = test_data.shape[1]

    # 2. 加载模型
    try:
        model = load_trained_model(model_name, config, device)
        if model is None: return
    except FileNotFoundError:
        print(f"Error: Model file for {model_name} not found.")
        return

    # 3. 执行预测
    preds, acts = make_predictions(
        model, test_data, config['seq_len'], config['pred_len'],
        device, model_name, df_transformer, stride=stride
    )

    n_samples, n_timesteps, n_features = preds.shape

    # 截断真实值数据
    if acts.shape[-1] > n_features:
        print(f"Truncating actuals: {acts.shape[-1]} -> {n_features} columns")
        acts = acts[:, :, :n_features]

    # --- 核心修改：在反归一化前，先计算归一化的 MSE ---
    norm_mse = mean_squared_error(acts.reshape(-1), preds.reshape(-1))

    print("Inverse transforming...")
    pred_orig = robust_inverse_transform(scaler, preds, n_samples, n_timesteps, n_features)
    act_orig = robust_inverse_transform(scaler, acts, n_samples, n_timesteps, n_features)

    # 5. 计算指标
    metrics = evaluate_metrics(pred_orig, act_orig)
    metrics['mse'] = norm_mse  # 替换为归一化 MSE

    print(f"\n{model_name} Metrics:")
    print(f"   Normalized MSE: {metrics['mse']:.4f}")
    print(f"   MAE:            {metrics['mae']:.4f}")
    print(f"   MAPE:           {metrics['mape']:.2f}%")
    print(f"   R2:             {metrics['r2']:.4f}")
    # 6. 保存结果
    save_dir = f'results/predictions/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to {save_dir} ...")

    test_start_idx = len(df_transformer) - len(test_data)
    first_pred_time_idx = test_start_idx + config['seq_len']
    end_idx = first_pred_time_idx + (n_samples * stride)

    full_index = df_transformer.index
    if end_idx > len(full_index):
        time_index = full_index[first_pred_time_idx:: stride][:n_samples]
    else:
        time_index = full_index[first_pred_time_idx: end_idx: stride]

    if len(time_index) != len(pred_orig):
        print(f"Warning: Index length mismatch ({len(time_index)} vs {len(pred_orig)}). Truncating.")
        min_len = min(len(time_index), len(pred_orig))
        time_index = time_index[:min_len]
        pred_orig = pred_orig[:min_len]
        act_orig = act_orig[:min_len]

    # 截断列名列表
    valid_feature_names = feature_names[:n_features]

    # 7. 导出 CSV
    export_df = pd.DataFrame(index=time_index)
    for i, fname in enumerate(valid_feature_names):
        export_df[f'{fname}_True'] = act_orig[:, 0, i]
        export_df[f'{fname}_Pred'] = pred_orig[:, 0, i]

    csv_path = f'{save_dir}/prediction_step1.csv'
    export_df.to_csv(csv_path)
    print(f"Prediction CSV saved: {csv_path}")

    # 8. 绘图
    plot_horizon_lines(pred_orig, act_orig, model_name, valid_feature_names)
    print(f"Plots generated.")