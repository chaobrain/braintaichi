import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 去除一组列中的异常值
def remove_row_outliers(df, column_prefix, threshold=1.5):
    time_columns = [col for col in df.columns if col.startswith(column_prefix)]
    
    for index, row in df.iterrows():
        time_values = row[time_columns]
        mean = time_values.mean()
        # 计算每个值与该行其他值的均值的差异
        deviation = time_values.sub(time_values.mean())
        abs_deviation = np.abs(deviation)
        # 标记那些与均值差异过大的值为异常值
        outliers = abs_deviation > (abs_deviation.mean() + threshold * abs_deviation.std())
        df.loc[index, time_columns] = time_values.mask(outliers, np.nan)

    return df

# 重新计算speedup
def calculate_speedup(df):
    # 计算每行taichi aot time与brainpy time的均值
    taichi_aot_time_columns = [col for col in df.columns if col.startswith('taichi aot time')]
    brainpy_time_columns = [col for col in df.columns if col.startswith('brainpy time')]
    df['taichi aot time mean'] = df[taichi_aot_time_columns].mean(axis=1)
    df['brainpy time mean'] = df[brainpy_time_columns].mean(axis=1)
    # 计算speedup
    df['speedup'] = np.where(df['brainpy time mean'] < df['taichi aot time mean'], 1 - (df['taichi aot time mean'] / df['brainpy time mean']), (df['brainpy time mean'] / df['taichi aot time mean']) - 1)
    return df

# Load the combined dataset
combined_df = pd.read_csv("event csr matvec/event_csrmv_gpu.csv")
combined_df.drop(columns=['backend'], inplace=True)
combined_df.drop(columns=['events type'], inplace=True)
combined_df.drop(columns=['s'], inplace=True)
combined_df.drop(columns=['p'], inplace=True)
combined_df = remove_row_outliers(combined_df, "taichi aot time")
combined_df = remove_row_outliers(combined_df, "brainpy time")
combined_df = calculate_speedup(combined_df)

plt.hist(combined_df['speedup'], bins=50)
plt.xlabel('Speedup')
plt.ylabel('Frequency')
plt.title('[GPU] event csr matvec taichi speedup over brainpylib' + '\n' +
          'speedup = (brainpylib time / taichi aot time) - 1')
plt.savefig('event csr matvec/event csrmv gpu dist.png')

