import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings

# 忽略警告
warnings.filterwarnings("ignore")


# ==========================================
# 1. 绘图风格配置 (Academic Style)
# ==========================================
def set_academic_style():
    # 先设置 seaborn 风格，再强制覆盖字体，防止被重置
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks", {"xtick.direction": "in", "ytick.direction": "in"})

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300


# ==========================================
# 2. 核心绘图函数 (完整版: 包含散点图和柱状图)
# ==========================================
def plot_full_analysis(load_series, temp_series, transformer_name, save_path, label_str):
    set_academic_style()

    # 数据清洗：去除空值
    mask = ~np.isnan(load_series) & ~np.isnan(temp_series)
    load_series = load_series[mask]
    temp_series = temp_series[mask]

    if len(load_series) == 0:
        print(f"Skipping {transformer_name}: No valid data.")
        return

    df = pd.DataFrame({'Temp': temp_series, 'Load': load_series})

    # 创建宽画布 (1x2 布局)
    fig = plt.figure(figsize=(15, 6.5))

    # -------------------------------------------------------
    # 左图: 非线性响应 (Hexbin + Fitting)
    # -------------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1)

    # Hexbin 密度图
    hb = ax1.hexbin(df['Temp'], df['Load'], gridsize=30, cmap='Blues', mincnt=1, edgecolors='none', alpha=0.9)
    cb = fig.colorbar(hb, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label('Sample Density', fontsize=12, fontweight='bold')

    # 拟合 U 型曲线 (2阶多项式)
    X_train = df[['Temp']].values
    y_train = df['Load'].values

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly, y_train)

    x_range = np.linspace(df['Temp'].min(), df['Temp'].max(), 100).reshape(-1, 1)
    y_pred = model.predict(poly.transform(x_range))

    # 拟合线使用 Nature 经典红色
    ax1.plot(x_range, y_pred, color='#EE0000', linewidth=3, linestyle='--', label='Nonlinear Trend ($2^{nd}$ Order)')

    ax1.set_xlabel('Temperature (°C)', fontweight='bold')
    ax1.set_ylabel('Load Magnitude (kW)', fontweight='bold')
    ax1.set_title('Load-Temp Response', y=-0.18, fontweight='bold', fontsize=14)
    ax1.legend(loc='upper center', frameon=True, fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # -------------------------------------------------------
    # 右图: 滞后相关性 (Lagged Correlation)
    # -------------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2)
    lags = [0, 1, 2, 3, 4, 6, 12, 24]
    corrs = []

    for lag in lags:
        if lag == 0:
            c = df['Load'].corr(df['Temp'])
        else:
            c = df['Load'].corr(df['Temp'].shift(lag))
        corrs.append(c)

    # 统一使用 Nature 经典学术蓝
    nature_blue = '#3B4992'
    bars = ax2.bar(range(len(lags)), corrs, color=nature_blue, alpha=0.85, width=0.6, edgecolor='black')

    # Y轴动态截断处理
    v_min, v_max = min(corrs), max(corrs)
    v_range = v_max - v_min

    if v_range == 0:
        bottom_limit = v_min * 0.9 if v_min > 0 else v_min * 1.1
        top_limit = v_max * 1.1
    else:
        # 底部留出 30% 极差的空白，顶部留出 60% 给数值标签
        margin = v_range * 0.3
        bottom_limit = v_min - margin
        if v_min > 0 and bottom_limit < 0:
            bottom_limit = 0
        top_limit = v_max + v_range * 0.6

    ax2.set_ylim(bottom_limit, top_limit)

    for bar in bars:
        height = bar.get_height()
        text_y_offset = height + (v_range * 0.05 if v_range > 0 else height * 0.01)

        # 保留4位小数
        ax2.text(bar.get_x() + bar.get_width() / 2., text_y_offset, f'{height:.4f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_xticks(range(len(lags)))
    ax2.set_xticklabels([f'{l}h' for l in lags])
    ax2.set_xlabel('Time Lag (Hours)', fontweight='bold')
    ax2.set_ylabel('Pearson Correlation (PCC)', fontweight='bold')
    ax2.set_title('Thermodynamic Inertia', y=-0.18, fontweight='bold', fontsize=14)

    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    # -------------------------------------------------------
    # 整体布局与 (a)(b)(c)(d) 标号
    # -------------------------------------------------------
    plt.tight_layout()
    # 在整张图片的底部正中间添加 (a)(b)(c)(d) 标号
    plt.figtext(0.5, -0.05, f'{label_str} Analysis of Transformer {transformer_name}',
                ha='center', fontsize=18, fontweight='bold')

    # 调整画布以容纳底部的总标题
    fig.subplots_adjust(bottom=0.1)

    # 保持保存在当前路径
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    print(f"✅ Figure generated and saved to: {save_path}")
    plt.close()


# ==========================================
# 3. 批量执行主程序
# ==========================================
def main():
    data_path = 'data/representative_data_with_weather.csv'

    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {data_path}")
        return

    all_cols = df.columns
    transformers = [c for c in all_cols if
                    'TEMP' not in c and 'date' not in c and 'DATETIME' not in c and 'Unnamed' not in c]

    # 提取前 4 个变压器
    target_transformers = transformers[:4]
    labels = ['(a)', '(b)', '(c)', '(d)']

    print(f"Selected 4 transformers for separate plots: {target_transformers}")

    for i, tr in enumerate(target_transformers):
        temp_col = f"TEMP_{tr}"

        if temp_col in df.columns:
            load_data = df[tr].values
            temp_data = df[temp_col].values

            # 保持存放在根目录
            save_name = f"Figure5_{tr}.jpg"
            plot_full_analysis(load_data, temp_data, tr, save_name, labels[i])
        else:
            print(f"⚠️ Warning: Temperature column '{temp_col}' not found for transformer '{tr}'")


if __name__ == "__main__":
    main()