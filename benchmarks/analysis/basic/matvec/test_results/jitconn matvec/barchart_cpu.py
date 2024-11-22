import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='C:\\Windows\\Fonts\\msyh.ttc')

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
PATH = os.path.dirname(os.path.abspath(__file__))
combined_df = pd.read_csv(os.path.join(PATH, 'jitconn_matvec_cpu.csv'))
combined_df.drop(columns=['backend'], inplace=True)
combined_df = remove_row_outliers(combined_df, "taichi aot time")
combined_df = remove_row_outliers(combined_df, "brainpy time")
combined_df = calculate_speedup(combined_df)

sns.set(style="whitegrid")
sns.histplot(combined_df['speedup'], bins=25, kde=True, color='darkorange')
# plt.hist(combined_df['speedup'], bins=50, density=True)

plt.xlabel('加速比', fontsize=16, fontproperties=font)
plt.ylabel('频次', fontsize=16, fontproperties=font)
# plt.title('[CPU] jitconn matvec taichi speedup over brainpylib')
plt.savefig(os.path.join(PATH, 'jitconn_matvec_cpu_dist.png'))


