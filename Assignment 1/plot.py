import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# read CSV data from 'timings.csv'
df = pd.read_csv('timings.csv')
df = df.sort_values(by='P')  # sort by process count for more visible plots

# plot settings
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

# create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

m_sizes = sorted(df['M'].unique())

# left plot - smaller M value
sns.boxplot(ax=axes[0], x='P', y='Time', data=df[df['M'] == m_sizes[0]], color='skyblue', width=0.5)
sns.stripplot(ax=axes[0], x='P', y='Time', data=df[df['M'] == m_sizes[0]], color='darkblue', size=6, jitter=True, alpha=0.7)
axes[0].set_title(f'Data Size M = {m_sizes[0]}', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Time (seconds)', fontsize=12)
axes[0].set_xlabel('Processes (P)', fontsize=12)

# right plot - larger M value
sns.boxplot(ax=axes[1], x='P', y='Time', data=df[df['M'] == m_sizes[1]], color='lightgreen', width=0.5)
sns.stripplot(ax=axes[1], x='P', y='Time', data=df[df['M'] == m_sizes[1]], color='darkgreen', size=6, jitter=True, alpha=0.7)
axes[1].set_title(f'Data Size M = {m_sizes[1]}', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Time (seconds)', fontsize=12)
axes[1].set_xlabel('Processes (P)', fontsize=12)

# legend for both plots
legend_elements_left = [
    Patch(facecolor='skyblue', edgecolor='black', label='Distribution'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=8, alpha=0.7, label='Individual runs')
]
legend_elements_right = [
    Patch(facecolor='lightgreen', edgecolor='black', label='Distribution'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', markersize=8, alpha=0.7, label='Individual runs')
]
axes[0].legend(handles=legend_elements_left, loc='upper left')
axes[1].legend(handles=legend_elements_right, loc='upper left')

plt.suptitle('Execution Time Analysis vs No. of Processes', fontsize=16, y=1.02)
plt.tight_layout()

# save output
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as plot.png")
plt.show()