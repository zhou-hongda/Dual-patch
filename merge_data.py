import pandas as pd
import numpy as np
import os


def merge_weather_correctly():
    print("开始执行数据融合 (使用2023年数据填补)...")

    # ================= 1. 读取数据 =================
    load_path = 'data/representative_data.csv'
    if not os.path.exists(load_path):
        print(f"错误: 找不到文件 {load_path}")
        return

    df_load = pd.read_csv(load_path)

    # 统一时间索引格式
    if 'DATETIME' in df_load.columns:
        df_load['DATETIME'] = pd.to_datetime(df_load['DATETIME'])
        df_load.set_index('DATETIME', inplace=True)
    else:
        df_load.iloc[:, 0] = pd.to_datetime(df_load.iloc[:, 0])
        df_load.set_index(df_load.columns[0], inplace=True)

    print(f"负载数据加载完成: {df_load.shape}")

    df_meta = pd.read_csv('data/transformer_meta.csv')
    df_weather = pd.read_csv('data/weather.csv')
    df_weather['DATETIME'] = pd.to_datetime(df_weather['DATETIME'])

    # ==========================================================
    # 数据填充逻辑：使用 2023 年同期数据填充 2021 年缺失段
    # ==========================================================

    # 定义缺失与源数据时间段
    missing_start = pd.Timestamp('2021-11-11')
    missing_end = pd.Timestamp('2021-12-31')

    # 往后找 2 年 (2021+2=2023)
    source_start = missing_start + pd.DateOffset(years=2)
    source_end = missing_end + pd.DateOffset(years=2)

    # 提取源数据
    mask_source = (df_weather['DATETIME'] >= source_start) & (df_weather['DATETIME'] <= source_end)
    df_fill = df_weather.loc[mask_source].copy()

    if not df_fill.empty:
        # 时间回推 2 年
        df_fill['DATETIME'] = df_fill['DATETIME'] - pd.DateOffset(years=2)

        # 合并并去重 (保留原有数据优先)
        df_weather = pd.concat([df_weather, df_fill], ignore_index=True)
        df_weather = df_weather.drop_duplicates(subset=['STATION_ID', 'DATETIME'], keep='first')

        print(f"已回填 {len(df_fill)} 条气象数据")
    else:
        print("未找到 2023 年同期数据，跳过填充")

    # ==========================================================

    # 预处理天气数据 (日均值)
    df_weather_clean = df_weather.groupby(['STATION_ID', 'DATETIME'])['TEMP'].mean()

    # ================= 2. 构建温度宽表 =================
    df_temps = pd.DataFrame(index=df_load.index)
    hourly_to_daily_map = pd.Series(df_load.index.floor('D'), index=df_load.index)
    transformers = df_load.columns

    print(f"正在为 {len(transformers)} 个变压器匹配温度...")

    for tid in transformers:
        meta_row = df_meta[df_meta['TRANSFORMER_ID'] == tid]

        # 如果找不到元数据，填0
        if meta_row.empty:
            df_temps[f'TEMP_{tid}'] = 0
            continue

        station_id = meta_row.iloc[0]['CLOSEST_STATION']

        try:
            # 获取对应站点温度
            daily_temps = df_weather_clean.xs(station_id, level='STATION_ID')

            # 广播到小时级
            mapped_temps = hourly_to_daily_map.map(daily_temps)

            # 缺失值修复
            if mapped_temps.isna().any():
                mapped_temps = mapped_temps.ffill().bfill().fillna(0)

            df_temps[f'TEMP_{tid}'] = mapped_temps

        except KeyError:
            df_temps[f'TEMP_{tid}'] = 0

    # ================= 3. 合并输出 =================
    df_final = pd.concat([df_load, df_temps], axis=1)

    output_path = 'data/representative_data_with_weather.csv'
    df_final.to_csv(output_path)
    print(f"处理完成，文件已保存: {output_path}")


if __name__ == "__main__":
    merge_weather_correctly()