import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
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

# ------------- identify runs -----------------
runs = experiment_data if isinstance(experiment_data, dict) else {}
dataset_name = "SPR_BENCH_or_toy"

# ----------- 1: loss curves ------------------
for run_idx, (run_name, run_data) in enumerate(runs.items()):
    if run_idx >= 5:
        break
    try:
        train_losses = run_data["losses"]["train"]
        val_losses = run_data["losses"]["val"]
        epochs = np.arange(1, len(train_losses) + 1)

        plt.figure()
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{run_name}: Train vs Validation Loss\nDataset: {dataset_name}")
        plt.legend()
        fname = os.path.join(working_dir, f"{run_name}_loss_curve_{dataset_name}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {run_name}: {e}")
        plt.close()

# ----------- 2: validation SWA ---------------
for run_idx, (run_name, run_data) in enumerate(runs.items()):
    if run_idx >= 5:
        break
    try:
        val_swa = run_data["metrics"]["val"]
        epochs = np.arange(1, len(val_swa) + 1)

        plt.figure()
        plt.plot(epochs, val_swa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.ylim(0, 1.05)
        plt.title(f"{run_name}: Validation SWA Over Epochs\nDataset: {dataset_name}")
        fname = os.path.join(working_dir, f"{run_name}_val_SWA_{dataset_name}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {run_name}: {e}")
        plt.close()

# ----------- 3: test SWA bar chart -----------
try:
    labels, swa_vals = [], []
    for run_name, run_data in runs.items():
        labels.append(run_name)
        swa_vals.append(run_data.get("metrics", {}).get("test", 0))

    x = np.arange(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar(x, swa_vals, width=0.5, color="skyblue")
    plt.xticks(x, labels, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title(f"Test SWA Across Runs\nDataset: {dataset_name}")
    fname = os.path.join(working_dir, f"test_SWA_comparison_{dataset_name}.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated SWA plot: {e}")
    plt.close()

# ------------ print metrics ------------------
print("Final Test SWA:")
for run_name, run_data in runs.items():
    swa = run_data.get("metrics", {}).get("test", 0)
    print(f"{run_name}: SWA = {swa:.4f}")
