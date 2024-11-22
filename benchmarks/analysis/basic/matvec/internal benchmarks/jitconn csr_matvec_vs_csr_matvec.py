import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__))

jitconn_matvec_cpu = f"{current_path}/jitconn_matvec_cpu.csv"
jitconn_matvec_gpu = f"{current_path}/jitconn_matvec_gpu.csv"
csrmv_cpu = f"{current_path}/csrmv_cpu.csv"
csrmv_gpu = f"{current_path}/csrmv_gpu.csv"
jitconn_matvec_grad_cpu = f"{current_path}/jitconn_matvec_grad_cpu.csv"
jitconn_matvec_grad_gpu = f"{current_path}/jitconn_matvec_grad_gpu.csv"
csrmv_grad_cpu = f"{current_path}/csrmv_grad_cpu.csv"
csrmv_grad_gpu = f"{current_path}/csrmv_grad_gpu.csv"

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
# Function to calculate the mean taichi aot time and speedup
def calculate_speedup(base_df, event_df):
    # Calculating mean taichi aot time for both datasets
    taichi_aot_time_columns = [col for col in base_df.columns if col.startswith('taichi aot time')]
    base_df['taichi aot time mean'] = base_df[taichi_aot_time_columns].mean(axis=1)
    event_df['taichi aot time mean'] = event_df[taichi_aot_time_columns].mean(axis=1)

    # Merging the datasets on shape[0], shape[1], values type, and transpose
    merged_df = pd.merge(base_df[['shape[0]', 'shape[1]', 'transpose', 'taichi aot time mean']],
                         event_df[['shape[0]', 'shape[1]', 'transpose', 'taichi aot time mean']],
                         on=['shape[0]', 'shape[1]', 'transpose'],
                         suffixes=('', '_event'))

    # Calculating speedup
    merged_df['speedup'] = np.where(merged_df['taichi aot time mean_event'] < merged_df['taichi aot time mean'],
                                    merged_df['taichi aot time mean_event'] / merged_df['taichi aot time mean'] - 1,
                                    1 - merged_df['taichi aot time mean'] / merged_df['taichi aot time mean_event'])

    return merged_df

# 计算这行的taichi aot time mean
def calculate_taichi_mean(df):
    taichi_aot_time_columns = [col for col in df.columns if col.startswith('taichi aot time')]
    df['taichi aot time mean'] = df[taichi_aot_time_columns].mean(axis=1)
    return df

def drop_columns(df):
    df.drop(columns=['backend'], inplace=True)
    return df

# Function to plot speedup heatmaps for different values type and transpose combinations
def plot_speedup_heatmaps(df, backend, filename):
    plt.figure(figsize=(14, 14))  # Adjusting figure size for multiple subplots

    # Creating subplots for each combination of values type and transpose
    for i, transpose in enumerate([tr  for tr in df['transpose'].unique()]):
        plt.subplot(2, 1, i + 1)

        # Filtering data and creating a pivot table for the heatmap
        filtered_df = df[(df['transpose'] == transpose)]
        heatmap_data = filtered_df.pivot_table(index='shape[0]', columns='shape[1]', values='speedup', aggfunc='mean')

        # Creating the heatmap
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='YlGnBu')
        plt.title(f'transpose: {transpose}')
        plt.xlabel('shape[1]')
        plt.ylabel('shape[0]')

    # Adding an overall title with adjusted position
    plt.suptitle(f'{backend} jitconn matvec speedup over csrmv)\n' + \
                 'speedup = common csrmv avg time / jitconn avg time - 1 if jitconn faster than common csrmv\n' + \
                 'speedup = 1 - jitconn avg time / common csrmv avg time if jitconn slower than common csrmv', fontsize=16)

    # Adjust layout to make room for the title
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    # plt.show()

# 读取合并csrmv_cpu.csv 和 event_csrmv_cpu.csv, csrmv_gpu.csv 和 event_csrmv_gpu.csv
# 只看values 和 transpose
jitconn_matvec_cpu_df = pd.read_csv(jitconn_matvec_cpu)
csrmv_cpu_df = pd.read_csv(csrmv_cpu)
jitconn_matvec_gpu_df = pd.read_csv(jitconn_matvec_gpu)
csrmv_gpu_df = pd.read_csv(csrmv_gpu)
jitconn_matvec_grad_cpu_df = pd.read_csv(jitconn_matvec_grad_cpu)
csrmv_grad_cpu_df = pd.read_csv(csrmv_grad_cpu)
jitconn_matvec_grad_gpu_df = pd.read_csv(jitconn_matvec_grad_gpu)
csrmv_grad_gpu_df = pd.read_csv(csrmv_grad_gpu)

jitconn_matvec_cpu_df = drop_columns(jitconn_matvec_cpu_df)
csrmv_cpu_df = drop_columns(csrmv_cpu_df)
jitconn_matvec_gpu_df = drop_columns(jitconn_matvec_gpu_df)
csrmv_gpu_df = drop_columns(csrmv_gpu_df)
jitconn_matvec_grad_cpu_df = drop_columns(jitconn_matvec_grad_cpu_df)
csrmv_grad_cpu_df = drop_columns(csrmv_grad_cpu_df)
jitconn_matvec_grad_gpu_df = drop_columns(jitconn_matvec_grad_gpu_df)
csrmv_grad_gpu_df = drop_columns(csrmv_grad_gpu_df)

jitconn_matvec_cpu_df = remove_row_outliers(jitconn_matvec_cpu_df, "taichi aot time")
csrmv_cpu_df = remove_row_outliers(csrmv_cpu_df, "taichi aot time")
jitconn_matvec_gpu_df = remove_row_outliers(jitconn_matvec_gpu_df, "taichi aot time")
csrmv_gpu_df = remove_row_outliers(csrmv_gpu_df, "taichi aot time")
jitconn_matvec_grad_cpu_df= remove_row_outliers(jitconn_matvec_grad_cpu_df, "taichi aot time")
csrmv_grad_cpu_df= remove_row_outliers(csrmv_grad_cpu_df, "taichi aot time")
jitconn_matvec_grad_gpu_df= remove_row_outliers(jitconn_matvec_grad_gpu_df, "taichi aot time")
csrmv_grad_gpu_df= remove_row_outliers(csrmv_grad_gpu_df, "taichi aot time")

speedup_cpu = calculate_speedup(jitconn_matvec_cpu_df, csrmv_cpu_df)
speedup_gpu = calculate_speedup(jitconn_matvec_gpu_df, csrmv_gpu_df)
speedup_grad_cpu = calculate_speedup(jitconn_matvec_grad_cpu_df, csrmv_grad_cpu_df)
speedup_grad_gpu = calculate_speedup(jitconn_matvec_grad_gpu_df, csrmv_grad_gpu_df)

# Plotting heatmaps for each dataset
plot_speedup_heatmaps(speedup_cpu, '[CPU] ', f'{current_path}/figure3/cpu.png')
plot_speedup_heatmaps(speedup_gpu, '[GPU] ', f'{current_path}/figure3/gpu.png')
plot_speedup_heatmaps(speedup_grad_cpu, '[CPU] Grad ', f'{current_path}/figure3/grad_cpu.png')
plot_speedup_heatmaps(speedup_grad_gpu, '[GPU] Grad ', f'{current_path}/figure3/grad_gpu.png')

