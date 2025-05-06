import pandas as pd
import matplotlib.pyplot as plt

# Set rcParams
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.2,
})

# Load the data
df = pd.read_csv("wandb_export_2025-05-06T11_16_29.783+01_00.csv")

# Covariance methods to plot (excluding 'sparse')
methods = [
    "full",
    "i.i.d",
    "block diagonal 12x12"
]

# Color cycle for plotting
colors = plt.cm.tab10.colors

# Plot setup
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each method
for idx, method in enumerate(methods):
    mean_col = f"covariance_method: {method} - validation/loss_mean"
    min_col = f"{mean_col}__MIN"
    max_col = f"{mean_col}__MAX"

    if mean_col in df.columns:
        # Drop rows where the mean is NaN
        subset = df[["Step", mean_col, min_col, max_col]].dropna()

        ax.plot(subset["Step"], subset[mean_col], label=method, color=colors[idx])
        ax.fill_between(subset["Step"], subset[min_col], subset[max_col],
                        color=colors[idx], alpha=0.2)

# Final touches
ax.set_xlabel("Iteration")
ax.set_ylabel("Validation Loss (Mean Reduction)")
ax.set_title("Validation During Training for Different Covariance Methods")
ax.set_xlim(0,1000)
ax.legend()
plt.tight_layout()
plt.savefig("loss_curves.pdf", dpi=300)
plt.show()
