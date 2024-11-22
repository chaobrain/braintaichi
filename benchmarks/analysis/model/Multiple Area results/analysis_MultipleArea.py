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


def main(benchmark_df):
    # Applying the functions
    benchmark_df = remove_outliers(benchmark_df, 'brainpylib_time')
    benchmark_df = remove_outliers(benchmark_df, 'taichi_time')
    benchmark_df = calculate_means_and_speedup(benchmark_df)
    DEVICE = benchmark_df['device'].unique()[0]
    benchmark_df = benchmark_df.drop('device', axis=1)

    # Aggregating speedup values
    agg_speedup_df = benchmark_df.groupby(['syn_func_type', 'neuron_num', 'conn_num']).mean().reset_index()

    # Unique communication types for creating separate heatmaps
    syn_func_types = agg_speedup_df['syn_func_type'].unique()

    # Setting up the plots
    plt.figure(figsize=(20, 20))

    for i, syn_func_type in enumerate(syn_func_types):
        plt.subplot(2, 2, i + 1)
        filtered_df = agg_speedup_df[agg_speedup_df['syn_func_type'] == syn_func_type]
        heatmap_data = filtered_df.pivot(index='neuron_num', columns='conn_num', values='speedup')

        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='YlGnBu', annot_kws={"size": 24})
        plt.title(f'[{DEVICE}]Multi area network Comm Type: {syn_func_type}\n' + \
                  'speedup = brainpylib time / taichi time - 1 if taichi faster than brainpylib\n' + \
                  'speedup = 1 - taichi time / brainpylib time if taichi slower than brainpylib')
        plt.xlabel('Conn Num')
        plt.ylabel('Neuron Num')

    plt.tight_layout()
    plt.savefig(f'{PATH}/benchmark__MultipleArea_{DEVICE}_heatmap.png')


# Load the dataset
benchmark_path = f"{PATH}/benchmark_MultipleArea_cpu.csv"
benchmark_df = pd.read_csv(benchmark_path)
main(benchmark_df)
benchmark_path = f"{PATH}/benchmark_MultipleArea_gpu.csv"
benchmark_df = pd.read_csv(benchmark_path)
main(benchmark_df)