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

runs = experiment_data.get("hidden_dim_tuning", {})
tags = sorted(runs.keys())[:5]  # safeguard max 5

# 1) Training & validation loss curves
try:
    plt.figure()
    for tag in tags:
        tr = runs[tag]["losses"]["train"]
        va = runs[tag]["losses"]["val"]
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"{tag}-train")
        plt.plot(epochs, va, "--", label=f"{tag}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Hidden_Dim_Tuning - Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "hidden_dim_tuning_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# 2) Validation HMWA curves
try:
    plt.figure()
    for tag in tags:
        hmwa = [m["hmwa"] for m in runs[tag]["metrics"]["val"]]
        plt.plot(range(1, len(hmwa) + 1), hmwa, label=tag)
    plt.xlabel("Epoch")
    plt.ylabel("HMWA")
    plt.title("Hidden_Dim_Tuning - Validation HMWA Across Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "hidden_dim_tuning_val_hmwa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val HMWA plot: {e}")
    plt.close()

# 3) Test HMWA bar chart
test_metrics = {}
try:
    for tag in tags:
        test_metrics[tag] = runs[tag]["metrics"]["test"]["hmwa"]
    plt.figure()
    plt.bar(
        range(len(test_metrics)),
        list(test_metrics.values()),
        tick_label=list(test_metrics.keys()),
    )
    plt.ylabel("Test HMWA")
    plt.title("Hidden_Dim_Tuning - Test HMWA by Hidden Size")
    fname = os.path.join(working_dir, "hidden_dim_tuning_test_hmwa_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test HMWA bar plot: {e}")
    plt.close()

# print metrics for quick inspection
for tag, hmwa in test_metrics.items():
    print(f"{tag} -> Test HMWA: {hmwa:.4f}")
