import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

combined_csv_path = "jitconn event matvec/jitconn_event_matvec_gpu.csv"

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

combined_df = pd.read_csv(combined_csv_path)
combined_df.drop(columns=['backend'], inplace=True)
combined_df = remove_row_outliers(combined_df, "taichi aot time")
combined_df = remove_row_outliers(combined_df, "brainpy time")
combined_df = calculate_speedup(combined_df)

# Averaging the speedup values for each combination of 's', 'p', 'backend', 'values type', and 'events type'
avg_speedup_df = combined_df.groupby(['type', 'transpose', 'outdim_parallel', 'bool_event', 'shape[0]', 'shape[1]']).mean().reset_index()

# Setting up the plots for heatmaps based on different backends, values types, and events types
plt.figure(figsize=(24, 22))  # Adjusting figure size for eight subplots

# Creating subplots for each combination of backend, values type, and events type
for i, (_type, transpose, bool_event, outdim_parallel) in enumerate([(_t, t, b, o)  for _t in avg_speedup_df['type'].unique() for t in avg_speedup_df['transpose'].unique() for b in avg_speedup_df['bool_event'].unique() for o in avg_speedup_df['outdim_parallel'].unique()]):
    plt.subplot(6, 4, i + 1)

    # Filtering data and creating a pivot table for the heatmap
    filtered_df = avg_speedup_df[(avg_speedup_df['type'] == _type) & (avg_speedup_df['transpose'] == transpose) & (avg_speedup_df['bool_event'] == bool_event) & (avg_speedup_df['outdim_parallel'] == outdim_parallel)]
    heatmap_data = filtered_df.pivot(index="shape[0]", columns="shape[1]", values="speedup")

    # Creating the heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='YlGnBu')
    plt.title(f'type: {_type}, transpose: {transpose}\n'
              f'bool_event: {bool_event}, outdim_parallel: {outdim_parallel}')
    plt.xlabel('shape[0]')
    plt.ylabel('shape[1]')

# Adding an overall title with adjusted position
plt.suptitle('[GPU] jitconn event matvec taichi speedup over brainpylib\n'
             'speedup = (brainpylib time / taichi aot time) - 1', fontsize=16)

# Adjust layout to make room for the title
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('jitconn event matvec/jitconn_event_matvec_gpu.png')
# plt.show()