import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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

def process_target_df(path):
    # Load the combined dataset
    combined_df = pd.read_csv(path)
    combined_df.drop(columns=['backend'], inplace=True)
    try:
        combined_df.drop(columns=['events type'], inplace=True)
        combined_df.drop(columns=['s'], inplace=True)
        combined_df.drop(columns=['p'], inplace=True)
    except:
        pass
    combined_df = remove_row_outliers(combined_df, "taichi aot time")
    combined_df = remove_row_outliers(combined_df, "brainpy time")
    combined_df = calculate_speedup(combined_df)
    return combined_df

def plot_stacked_histograms_with_fixed_bins(case_dict, root_path, num_bins=40):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))  # 创建两个子图
    colors = ['blue', 'green', 'red', 'cyan']  # 定义每个案例的颜色

    for i, (device, cases) in enumerate(case_dict.items()):
        all_speedups = []  # 收集所有speedup值以计算共享的bins范围
        for case_name, file_path in cases.items():
            data_path = os.path.join(root_path, file_path)
            df = process_target_df(data_path)  # 加载和处理数据
            all_speedups.extend(df['speedup'].values)  # 添加到速度提升列表

        # 基于当前设备的所有speedup值计算bins
        min_speedup, max_speedup = min(all_speedups), max(all_speedups)
        bins = np.linspace(min_speedup, max_speedup, num_bins + 1)
        bottom = np.zeros(len(bins)-1)  # 初始化底部累积值

        for j, (case_name, file_path) in enumerate(cases.items()):
            data_path = os.path.join(root_path, file_path)
            df = process_target_df(data_path)  # 再次加载和处理数据
            # 计算当前案例的直方图
            counts, _ = np.histogram(df['speedup'], bins=bins, density=True)
            
            # 绘制柱状图
            axs[i].bar(bins[:-1] + np.diff(bins)/2, counts, width=np.diff(bins), bottom=bottom, color=colors[j], label=case_name, alpha=0.75)
            bottom += counts  # 更新底部累积值，为下一个案例做准备

        axs[i].set_title(f'{device} backend')
        axs[i].set_xlabel('Speedup')
        axs[i].set_ylabel('Density')
        axs[i].legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(root_path, 'stacked_histograms.png'))




# Load the combined dataset
PATH = os.path.dirname(os.path.abspath(__file__))
case_dict = {
    'CPU': {
        'csr matvec': os.path.join(PATH, 'csr matvec', 'csrmv_cpu.csv'),
        'event csr matvec': os.path.join(PATH, 'event csr matvec', 'event_csrmv_cpu.csv'),
        'jitconn matvec': os.path.join(PATH, 'jitconn matvec', 'jitconn_matvec_cpu.csv'),
        'jitconn event matvec': os.path.join(PATH, 'jitconn event matvec', 'jitconn_event_matvec_cpu.csv'),
    },
    'GPU': {
        'csr matvec': os.path.join(PATH, 'csr matvec', 'csrmv_gpu.csv'),
        'event csr matvec': os.path.join(PATH, 'event csr matvec', 'event_csrmv_gpu.csv'),
        'jitconn matvec': os.path.join(PATH, 'jitconn matvec', 'jitconn_matvec_gpu.csv'),
        'jitconn event matvec': os.path.join(PATH, 'jitconn event matvec', 'jitconn_event_matvec_gpu.csv'),
    },
}

plot_stacked_histograms_with_fixed_bins(case_dict, PATH)