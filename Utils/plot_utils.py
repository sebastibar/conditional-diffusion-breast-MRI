import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_curves(all_metrics, loss_components):
    """
    Plots metrics and loss components after training.
    Args:
        all_metrics: List of dictionaries with epoch metrics ({'Epoch', 'MAE', 'MSE'})
        loss_components: Dictionary with loss components per epoch
    """
    metrics_df = pd.DataFrame(all_metrics)
    loss_df = pd.DataFrame(loss_components)

    # === 1. Normalized Metric Trends ===
    normalized = metrics_df.copy()
    for col in ['MAE', 'MSE']:
        min_val = metrics_df[col].min()
        max_val = metrics_df[col].max()
        normalized[col] = (metrics_df[col] - min_val) / (max_val - min_val + 1e-8)

    plt.figure(figsize=(10, 5))
    for col in ['MAE', 'MSE']:
        sns.lineplot(data=normalized, x="Epoch", y=col, label=col)
    plt.title("Normalized Metric Trends Over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Normalized Value (0–1)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # === 2. Raw Metrics Subplots ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    cols = ['MAE', 'MSE']
    for i, col in enumerate(cols):
        sns.lineplot(data=metrics_df, x="Epoch", y=col, ax=axs[i])
        axs[i].set_title(col); axs[i].set_xlabel("Epoch"); axs[i].set_ylabel(col); axs[i].grid(True)
    plt.suptitle("Metric Evolution Per Epoch (Raw Scale)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # === 3. Loss Components ===
    plt.figure(figsize=(12, 6))
    for col in ["MAE", "Percep", "TV", "MSE", "Total"]:
        if col in loss_df.columns: # Check if key exists
            sns.lineplot(data=loss_df, x="Epoch", y=col, label=col)
    plt.title("Loss Components Over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()
