import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

PATH = os.path.dirname(os.path.abspath(__file__))

# Load the data from CSV file
data = pd.read_csv(os.path.join(PATH, 'benchmark_COBA_spks_cpu.csv'))
# Calculate the absolute difference between brainpylib_spk and taichi_spk
data['abs_diff'] = np.abs(data['brainpylib_spk'] - data['taichi_spk'])

# Apply a non-linear (logarithmic) transformation to the 'post_num' column
data['post_num_log'] = np.log10(data['post_num'])

# Plotting with a regression line to show the trend
plt.figure(figsize=(12, 6))
sns.regplot(x='post_num_log', y='abs_diff', data=data, scatter=True, color='skyblue')
# plt.title('Trend of Absolute Difference of Spikes with Regression (CPU)')
plt.xlabel('Log10 of Total Neurons')
plt.ylabel('Absolute Difference of Spikes per Neuron per Second')
plt.ylim(0, None)  # Set the lower limit of y-axis to 0
# plt.grid(True)
plt.savefig(os.path.join(PATH, 'benchmark_COBA_spks_cpu.png'))
plt.show()