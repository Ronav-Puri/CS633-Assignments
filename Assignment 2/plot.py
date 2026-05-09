import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
file_path = "final_timing.csv"
df = pd.read_csv(file_path)

# Clean column names
df.columns = [col.strip() for col in df.columns]

# Sort
df = df.sort_values(by=["P", "N"])

# Unique values
processes = sorted(df["P"].unique())
data_sizes = sorted(df["N"].unique())

# Create subplots
fig, axes = plt.subplots(1, len(data_sizes), figsize=(14, 6), sharey=False)

if len(data_sizes) == 1:
    axes = [axes]

print("\n===== STATISTICS =====\n")

for idx, n in enumerate(data_sizes):
    ax = axes[idx]
    subset_n = df[df["N"] == n]

    positions = []
    box_data = []
    labels = []

    pos = 1

    for p in processes:
        subset = subset_n[subset_n["P"] == p]

        if not subset.empty:
            times = subset["Time"].values

            # Compute the stats used for analysing
            median = np.median(times)
            std = np.std(times)

            # Print the Stats (the median and the standard deviation)
            print(f"P={p}, N={n} -> Median: {median:.6f}, Std Dev: {std:.6f}")

            box_data.append(times)
            positions.append(pos)
            labels.append(f"P={p}")

            pos += 1

    # Boxplot
    box = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True
    )

    # Color
    for patch in box['boxes']:
        patch.set_facecolor("lightblue")

    # Labels
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(f"N = {n}")
    ax.set_xlabel("Processes (P)")
    ax.set_ylabel("Time (seconds)")

    # Grid
    ax.grid(axis="y", linestyle="--", alpha=0.7)

# Title
plt.suptitle("Execution Time Distribution by Process Count (Split by Data Size)")

plt.tight_layout()
plt.savefig("execution_time_boxplot.png", dpi=300)
plt.show()