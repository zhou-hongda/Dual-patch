import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

# 设置中文和绘图风格
# 设置英文和绘图风格 (替换原有的 SimHei 设置)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid", rc={"font.family": "serif", "font.serif": ["Times New Roman"]})


# =================================================================
# 1. 数据加载与清洗 (返回未填充的带 NaN 的数据框)
# =================================================================

def load_and_pivot_data(file_path):
    """
    加载原始数据并转换为矩阵形式 (不进行插值填充)
    """
    print(f"正在加载数据: {file_path} ...")
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("不支持的文件格式")
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到，请检查路径: {file_path}")
        raise

    if 'TRANSFORMER_ID' not in df.columns or 'LOAD' not in df.columns:
        raise ValueError("数据缺少 TRANSFORMER_ID 或 LOAD 列")

    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.drop_duplicates(subset=['TRANSFORMER_ID', 'DATETIME'])

    # 强制清洗：确保 LOAD 列是数值类型，非数值内容变成 np.nan
    if df['LOAD'].dtype == 'object':
        df['LOAD'] = df['LOAD'].astype(str).str.strip().replace('', np.nan)

    df['LOAD'] = pd.to_numeric(df['LOAD'], errors='coerce')

    # 剔除因为强制转换而变成 NaN 的整行原始记录
    df = df.dropna(subset=['LOAD'])

    print("正在重组数据结构 (Pivot)...")
    pivot_df = df.pivot(index='TRANSFORMER_ID', columns='DATETIME', values='LOAD')

    # 首次剔除：剔除所有数据点全为空的变压器 ID
    rows_before_drop = len(pivot_df)
    pivot_df = pivot_df.dropna(axis=0, how='all')
    rows_dropped = rows_before_drop - len(pivot_df)
    if rows_dropped > 0:
        print(f"❌ 警告: 剔除了 {rows_dropped} 个无任何负载记录的变压器。")

    return pivot_df


# =================================================================
# 2. 数据过滤 (在未填充的数据上执行)
# =================================================================

def filter_data(pivot_df,
                min_mean_load=10.0,
                max_inactive_ratio=0.4,
                inactive_threshold=5.0,
                max_initial_nan_ratio=0.2):
    """
    综合过滤函数，在带原始 NaN 的 pivot_df 上执行过滤
    """
    print(f"\n正在执行数据清洗...")
    original_count = len(pivot_df)

    # 1. 【核心修复】检查序列起始连续 NaN 的比例 (解决类别4问题)
    n_cols = pivot_df.shape[1]
    max_nan_count = int(n_cols * max_initial_nan_ratio)

    # 统计每个变压器从序列开头开始连续 NaN 的数量
    initial_nan_counts = pivot_df.isnull().cumsum(axis=1).eq(np.arange(1, n_cols + 1)).sum(axis=1)

    # 只有起始连续 NaN 数量少于阈值的才保留
    mask_start = initial_nan_counts <= max_nan_count
    print(f"  - 规则1: 序列起始连续 NaN 比例必须 < {max_initial_nan_ratio * 100}%")

    # 2. 过滤平均负载过小的 (使用 fillna(0) 应对 NaN)
    # fillna(0) 确保 mean() 不会因为整行是 NaN 而失败
    mean_loads = pivot_df.mean(axis=1)
    mask_size = mean_loads.fillna(0) > min_mean_load
    print(f"  - 规则2: 平均负载必须 > {min_mean_load}")

    # 3. 过滤长期"不活跃"的变压器 (负载长期接近 0)
    # 使用 fillna(0) 应对 NaN
    is_inactive = pivot_df.fillna(0) <= inactive_threshold
    inactive_ratios = is_inactive.mean(axis=1)
    mask_activity = inactive_ratios < max_inactive_ratio
    print(f"  - 规则3: '不活跃'时间占比必须 < {max_inactive_ratio * 100}% (阈值: {inactive_threshold})")

    # 综合所有条件
    final_mask = mask_start & mask_size & mask_activity
    filtered_df = pivot_df[final_mask]

    # 打印详细的剔除原因统计
    print("-" * 30)
    print(f"因[前期未投运/NaN过多]剔除: {(~mask_start).sum()} 个")
    print(f"因[平均负载太小]剔除: {(~mask_size).sum()} 个")
    print(f"因[长期不活跃]剔除:   {(~mask_activity).sum()} 个")
    print("-" * 30)
    print(f"❌ 总计剔除: {original_count - len(filtered_df)} 个")
    print(f"✅ 最终保留: {len(filtered_df)} 个变压器")

    if len(filtered_df) == 0:
        raise ValueError("所有数据都被过滤了！请放宽过滤条件。")

    return filtered_df


# =================================================================
# 3. 数据插值与标准化
# =================================================================

def fill_data(pivot_df_with_nan):
    """
    对过滤后的数据进行插值和填充
    """
    print("\n正在对过滤后的数据进行插值与填充...")
    # 线性插值（解决中间缺失点）
    filled_df = pivot_df_with_nan.interpolate(method='linear', axis=1)

    # 极值填充（解决首尾缺失点），这一步会将起始 NaN 变成平直线
    filled_df = filled_df.bfill(axis=1).ffill(axis=1)

    # 最终检查，剔除无法填充的稀疏数据
    filled_df = filled_df.dropna(axis=0, how='any')

    return filled_df


def normalize_data(pivot_df):
    """Z-Score 标准化"""
    scaler = StandardScaler()
    # 对每一行（变压器）进行标准化
    data_scaled = scaler.fit_transform(pivot_df.T).T
    return data_scaled, scaler


# =================================================================
# 4. K 值确定与聚类执行
# =================================================================

def calculate_elbow_point(data_scaled, max_k=12):
    """手肘法自动寻找最佳 K 值 (全英文版)"""
    print(f"\nExecuting Elbow Method analysis (Testing K=2 to {max_k})...")
    inertias = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_scaled)
        inertias.append(kmeans.inertia_)

    # 自动计算手肘点 (最大距离法)
    x1, y1 = K_range[0], inertias[0]
    x2, y2 = K_range[-1], inertias[-1]

    distances = []
    for i in range(len(K_range)):
        x0 = K_range[i]
        y0 = inertias[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        distances.append(numerator / denominator)

    best_idx = np.argmax(distances)
    optimal_k = K_range[best_idx]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', markersize=8)
    plt.plot(optimal_k, inertias[best_idx], 'r*', markersize=15, label=f'Optimal K={optimal_k}')
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=14)
    plt.title('Elbow Method', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.5)
    plt.savefig('results/elbow_analysis.png', dpi=300, bbox_inches='tight')

    print(f"💡 Recommended optimal clusters K = {optimal_k}")
    return optimal_k



def perform_clustering(data_scaled, transformer_ids, n_clusters, top_n=1):
    """
    【核心修改】基于“距离+相关性”混合打分筛选代表
    旨在选出“形状最典型”而非仅仅“数值平均”的变压器
    """
    print(f"\n🚀 开始执行高精度筛选 (K={n_clusters})...")

    # 1. 执行基础 K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    centers = kmeans.cluster_centers_

    # 计算欧氏距离 (数值接近度)
    # shape: (n_samples, n_clusters)
    dist_matrix = kmeans.transform(data_scaled)

    representative_ids_all = []

    print("\n=== 筛选逻辑: 综合考量 [欧氏距离] 与 [形状相关性] ===")

    for i in range(n_clusters):
        # --- 步骤 A: 获取该类成员 ---
        cluster_indices = np.where(labels == i)[0]
        cluster_data = data_scaled[cluster_indices]
        cluster_ids = transformer_ids[cluster_indices]
        center_curve = centers[i]

        # 如果类内只有一个成员，直接选中
        if len(cluster_indices) <= 1:
            representative_ids_all.extend(cluster_ids.tolist())
            continue

        # --- 步骤 B: 计算混合得分 ---
        scores = []

        # 获取该类成员到中心的欧氏距离
        dists = dist_matrix[cluster_indices, i]

        # 归一化距离 (越小越好 -> 转为 0-1 之间，越大越好)
        # 避免除以0
        max_dist = dists.max() if dists.max() > 0 else 1.0
        dist_scores = 1 - (dists / max_dist)  # 1.0 表示距离最近(最好)

        # 计算相关性 (形状相似度)
        corr_scores = []
        for row in cluster_data:
            # 计算当前变压器与聚类中心的皮尔逊相关系数
            # [0] 是相关系数，[1] 是 p-value
            corr = pearsonr(row, center_curve)[0]
            # 填充 NaN (如果数据完全由0组成可能导致NaN)
            if np.isnan(corr): corr = -1
            corr_scores.append(corr)
        corr_scores = np.array(corr_scores)

        # === 核心公式 ===
        # 权重分配：形状(Correlation)占 70%，数值距离(Distance)占 30%
        # 理由：在时序预测中，趋势的一致性比绝对值的接近更重要，因为绝对值会被 Scaler 处理掉
        final_scores = 0.7 * corr_scores + 0.3 * dist_scores

        # --- 步骤 C: 排序取 Top-N ---
        # argsort 是从小到大，所以取负号或者[::-1]反转
        sorted_indices_local = np.argsort(final_scores)[::-1][:top_n]

        best_ids = cluster_ids[sorted_indices_local].tolist()
        best_scores = final_scores[sorted_indices_local]

        representative_ids_all.extend(best_ids)

        print(f"类别 {i + 1}: 成员数 {len(cluster_ids)}")
        for bid, bscore in zip(best_ids, best_scores):
            print(f"   -> 选中: {bid} (综合得分: {bscore:.4f})")

    return representative_ids_all, labels, centers

# =================================================================
# 5. 可视化
# =================================================================

def plot_clusters(pivot_df, data_scaled, labels, centers, representative_ids, n_clusters):
    """可视化聚类结果 (全英文版)"""
    print("\nGenerating clustering visualization...")
    rows = n_clusters
    fig, axes = plt.subplots(rows, 1, figsize=(15, 3 * rows), sharex=True)
    if n_clusters == 1: axes = [axes]

    time_points = pivot_df.columns
    plot_step = max(1, len(time_points) // 200)

    for i in range(n_clusters):
        ax = axes[i]
        cluster_indices = np.where(labels == i)[0]

        # 绘制成员(灰)
        for idx in cluster_indices:
            ax.plot(time_points[::plot_step], data_scaled[idx, ::plot_step],
                    color='gray', alpha=0.05, linewidth=0.5)

        # 绘制中心(红)
        ax.plot(time_points[::plot_step], centers[i, ::plot_step],
                color='red', linestyle='--', linewidth=2, label='Cluster Center')

        # 绘制代表(蓝)
        rep_id = representative_ids[i]
        rep_idx = np.where(pivot_df.index == rep_id)[0][0]
        ax.plot(time_points[::plot_step], data_scaled[rep_idx, ::plot_step],
                color='blue', linewidth=1.5, label=f'Representative ({rep_id})')

        ax.set_title(f"Cluster {i + 1} (Size: {len(cluster_indices)})", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('results/clustering_result.png', dpi=300, bbox_inches='tight')
    print("📊 Clustering results saved to: results/clustering_result.png")
# =================================================================
# 6. 主函数
# =================================================================

def main():
    import os
    os.makedirs('results', exist_ok=True)

    # === 参数配置区 (请根据实际数据调整) ===
    file_path = 'data/transformer_raw.csv'

    # 1. 最小平均负载 (过滤太小的)
    min_mean_load = 10.0

    # 2. 最大不活跃比例 (过滤长期低负载的)
    max_inactive_ratio = 0.4  # 允许 40% 的时间不活跃

    # 3. 不活跃阈值 (负载低于此值视为不活跃)
    inactive_threshold = 5.0  # 负载 < 5.0 kW 视为不活跃

    # 4. 【关键】允许起始 NaN 的最大比例 (解决前期缺失问题)
    max_initial_nan_ratio = 0.2  # 允许前 20% 的时间是 NaN

    try:
        # 1. 加载和透视 (返回带 NaN 的 DF)
        pivot_df_with_nan = load_and_pivot_data(file_path)

        # 2. 过滤 (在带 NaN 的 DF 上执行所有清洗规则)
        filtered_df_with_nan = filter_data(pivot_df_with_nan,
                                           min_mean_load=min_mean_load,
                                           max_inactive_ratio=max_inactive_ratio,
                                           inactive_threshold=inactive_threshold,
                                           max_initial_nan_ratio=max_initial_nan_ratio)

        # 3. 填充 (对过滤后的干净数据进行插值填充)
        pivot_df = fill_data(filtered_df_with_nan)
        transformer_ids = pivot_df.index

        # 4. 标准化
        data_scaled, _ = normalize_data(pivot_df)

        # 5. 确定 K 值
        optimal_k = calculate_elbow_point(data_scaled, max_k=12)

        # 6. 执行聚类
        representatives, labels, centers = perform_clustering(data_scaled, transformer_ids, optimal_k)

        # 7. 可视化
        plot_clusters(pivot_df, data_scaled, labels, centers, representatives, optimal_k)

        # 8. 保存结果
        print("\n" + "=" * 50)
        print(f"🎉 最终提取的 {len(representatives)} 个代表变压器 ID：")
        print(representatives)
        print("=" * 50)

        with open('results/representative_transformers.txt', 'w') as f:
            for rid in representatives:
                f.write(f"{rid}\n")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 发生致命错误，请检查配置参数和原始数据文件: {e}")


if __name__ == "__main__":
    main()