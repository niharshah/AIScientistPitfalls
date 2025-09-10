import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

root = experiment_data.get("multi_synth", {})
datasets = list(root.keys())
epochs = len(next(iter(root.values()))["losses"]["train"]) if root else 0
ep_range = list(range(1, epochs + 1))

# 1) losses
try:
    plt.figure()
    for name in datasets:
        tr = root[name]["losses"]["train"]
        vl = root[name]["losses"]["val"]
        if tr and vl:
            plt.plot(ep_range, tr, label=f"{name}-train")
            plt.plot(ep_range, vl, label=f"{name}-val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Losses - multi_synth")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "multi_synth_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) validation metrics over epochs
try:
    metrics_list = ["CWA", "SWA", "GCWA"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, m in enumerate(metrics_list):
        ax = axes[idx]
        for name in datasets:
            vals = [d.get(m, np.nan) for d in root[name]["metrics"]["val"]]
            ax.plot(ep_range, vals, label=name)
        ax.set_title(m)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(m)
        ax.legend()
    fig.suptitle("Validation Metrics over Epochs - multi_synth")
    fig.tight_layout()
    plt.savefig(os.path.join(working_dir, "multi_synth_val_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation metric plot: {e}")
    plt.close()

# 3) test metrics bar chart
try:
    metrics_list = ["CWA", "SWA", "GCWA"]
    x = np.arange(len(datasets))
    width = 0.25
    plt.figure(figsize=(8, 4))
    for i, m in enumerate(metrics_list):
        vals = [root[name]["metrics"]["test"].get(m, np.nan) for name in datasets]
        plt.bar(x + i * width - width, vals, width, label=m)
    plt.xticks(x, datasets)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Test Metrics - multi_synth")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "multi_synth_test_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test metric plot: {e}")
    plt.close()
