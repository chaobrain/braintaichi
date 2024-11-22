import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

PATH = os.path.dirname(os.path.abspath(__file__))

# Function to remove outliers for brainpylib and taichi execution times
def remove_outliers(df, prefix):
    time_columns = [col for col in df.columns if col.startswith(prefix)]
    for index, row in df.iterrows():
        time_values = row[time_columns]
        mean = time_values.mean()
        deviation = time_values.sub(mean)
        abs_deviation = abs(deviation)
        outliers = abs_deviation > (abs_deviation.mean() + 1.5 * abs_deviation.std())
        df.loc[index, time_columns] = time_values.mask(outliers, np.nan)
    return df

# Calculate mean execution times and speedup
def calculate_means_and_speedup(df):
    brainpylib_columns = [col for col in df.columns if col.startswith('brainpylib_time')]
    taichi_columns = [col for col in df.columns if col.startswith('taichi_time')]
    df['brainpylib_mean'] = df[brainpylib_columns].mean(axis=1)
    df['taichi_mean'] = df[taichi_columns].mean(axis=1)
    df['speedup'] = np.where(df['taichi_mean'] > df['brainpylib_mean'],
                            1 - (df['taichi_mean'] / df['brainpylib_mean']),
                            (df['brainpylib_mean'] / df['taichi_mean']) - 1)
    return df


def process_target_df(path):
    # Load the combined dataset
    benchmark_df = pd.read_csv(path)
    # Applying the functions
    benchmark_df = remove_outliers(benchmark_df, 'brainpylib_time')
    benchmark_df = remove_outliers(benchmark_df, 'taichi_time')
    benchmark_df = calculate_means_and_speedup(benchmark_df)
    DEVICE = benchmark_df['device'].unique()[0]
    benchmark_df = benchmark_df.drop('device', axis=1)

    # Aggregating speedup values
    agg_speedup_df = benchmark_df.groupby(['comm_type', 'post_num', 'conn_num']).mean().reset_index()

    return agg_speedup_df

def plot_stacked_histograms_with_fixed_bins(root_path, num_bins=40):
    plt.figure(figsize=(12, 10))  # 调整为单个图形
    colors = ['blue', 'green', 'red', 'cyan']  # 定义每个案例的颜色

    cpu_path = os.path.join(root_path, 'benchmark_COBA_cpu.csv')
    gpu_path = os.path.join(root_path, 'benchmark_COBA_gpu.csv')
    cpu_df = process_target_df(cpu_path)
    gpu_df = process_target_df(gpu_path)
    devices_df = {'CPU': cpu_df, 'GPU': gpu_df}

    # 为CPU和GPU绘制不同的图
    for i, (device, df) in enumerate(devices_df.items()):
        plt.subplot(2, 1, i + 1)  # 创建子图
        bottom = np.zeros(num_bins)  # 重置底部数组

        # 按comm_type分组绘制
        for j, comm_type in enumerate(df['comm_type'].unique()):
            comm_df = df[df['comm_type'] == comm_type]
            counts, bins = np.histogram(comm_df['speedup'], bins=num_bins, range=(df['speedup'].min(), df['speedup'].max()), density=True)
            plt.bar(bins[:-1], counts, width=(bins[1] - bins[0]), bottom=bottom, color=colors[j], alpha=0.7, label=comm_type)
            bottom += counts  # 更新底部以堆叠下一组

        plt.title(f'COBA network [{device}]')
        plt.xlabel('Speedup')
        plt.ylabel('Density')
        plt.legend(title='Comm Type')

    plt.tight_layout()
    plt.savefig(f'{root_path}/benchmark_COBA_speedup_stacked.png')
    plt.show()



# Load the dataset
PATH = os.path.dirname(os.path.abspath(__file__))
case_dict = {
    'CPU': [
        # 'EventCSRLinear',
        'EventJitFPHomoLinear',
        # 'EventJitFPUniformLinear',
        # 'EventJitFPNormalLinear',
    ],
    'GPU': [
        # 'EventCSRLinear',
        'EventJitFPHomoLinear',
        # 'EventJitFPUniformLinear',
        # 'EventJitFPNormalLinear',
    ],
}
plot_stacked_histograms_with_fixed_bins(PATH)
