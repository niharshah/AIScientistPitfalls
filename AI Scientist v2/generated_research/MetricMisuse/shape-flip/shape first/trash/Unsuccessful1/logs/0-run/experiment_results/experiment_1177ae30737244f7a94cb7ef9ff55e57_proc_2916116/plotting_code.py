import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure working directory exists
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------
dataset = "spr_bench"
if dataset in experiment_data:
    data = experiment_data[dataset]
    train_losses = data["losses"]["train"]
    val_losses = data["losses"]["val"]
    val_metrics = data["metrics"]["val"]
    test_metrics = data["metrics"]["test"]
    epochs = np.arange(1, len(train_losses) + 1)
else:
    print(f"Dataset {dataset} not found in experiment data.")
    train_losses = val_losses = val_metrics = []
    test_metrics = {}

# ------------------------------------------------------------------
# 1) Loss curves ----------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR-Bench Loss Curves\nLeft: Training, Right: Validation")
    plt.legend()
    save_path = os.path.join(working_dir, f"{dataset}_loss_curves.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Validation metrics over epochs --------------------------------
try:
    if val_metrics:
        swa = [m["swa"] for m in val_metrics]
        cwa = [m["cwa"] for m in val_metrics]
        hwa = [m["hwa"] for m in val_metrics]
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, hwa, label="HWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR-Bench Validation Metrics\nLeft: SWA, Middle: CWA, Right: HWA")
        plt.legend()
        save_path = os.path.join(working_dir, f"{dataset}_val_metrics.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()

# 3) Test metrics bar chart ----------------------------------------
try:
    if test_metrics:
        plt.figure()
        labels = ["SWA", "CWA", "HWA"]
        values = [test_metrics[k.lower()] for k in labels]
        plt.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
        plt.ylim(0, 1)
        plt.title("SPR-Bench Test Metrics\nBars: SWA, CWA, HWA")
        save_path = os.path.join(working_dir, f"{dataset}_test_metrics.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print final test metrics
if test_metrics:
    print("Final Test Metrics:", test_metrics)
