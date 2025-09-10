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

runs = experiment_data.get("runs", {})
max_figs = 5
fig_count = 0

# ---------- 1-4: loss curves ----------
for run_idx, (run_name, run_data) in enumerate(runs.items()):
    if fig_count >= max_figs or run_idx >= 4:
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
        plt.title(f"{run_name}: Training vs Validation Loss\nDataset: SPR_BENCH (toy)")
        plt.legend()
        fname = os.path.join(working_dir, f"{run_name}_loss_SPR_BENCH.png")
        plt.savefig(fname)
        plt.close()
        fig_count += 1
    except Exception as e:
        print(f"Error creating loss plot for {run_name}: {e}")
        plt.close()

# ---------- 5: aggregated SWA ----------
if fig_count < max_figs:
    try:
        labels, swa_vals = [], []
        for run_name, run_data in runs.items():
            labels.append(run_name)
            swa_vals.append(run_data.get("metrics", {}).get("test", 0))

        x = np.arange(len(labels))
        plt.figure(figsize=(8, 4))
        plt.bar(x, swa_vals, width=0.6, color="skyblue", label="SWA")
        plt.xticks(x, labels, rotation=15)
        plt.ylim(0, 1.05)
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("Test SWA Across Model Variants\nDataset: SPR_BENCH (toy)")
        plt.legend()
        fname = os.path.join(working_dir, "test_SWA_comparison_SPR_BENCH.png")
        plt.savefig(fname)
        plt.close()
        fig_count += 1
    except Exception as e:
        print(f"Error creating aggregated SWA plot: {e}")
        plt.close()

# ---------- print evaluation metrics ----------
print("\nFinal Test Metrics (SWA):")
for run_name, run_data in runs.items():
    swa = run_data.get("metrics", {}).get("test", 0)
    print(f"{run_name}: SWA={swa:.4f}")
