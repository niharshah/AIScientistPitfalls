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

# -------------------- per-run loss & SWA curves --------------------
max_figs = 4  # keep total <=5 incl. summary
for i, (run_name, run_data) in enumerate(runs.items()):
    if i >= max_figs:
        break
    try:
        train_losses = np.asarray(run_data["losses"]["train"])
        val_losses = np.asarray(run_data["losses"]["val"])
        swa_vals = np.asarray(run_data["SWA"]["val"])
        epochs = np.arange(1, len(train_losses) + 1)

        plt.figure(figsize=(6, 4))
        ax1 = plt.gca()
        ax1.plot(epochs, train_losses, label="Train Loss", color="tab:blue")
        ax1.plot(epochs, val_losses, label="Val Loss", color="tab:orange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross-Entropy Loss")
        ax2 = ax1.twinx()
        ax2.plot(epochs, swa_vals, label="Val SWA", color="tab:green")
        ax2.set_ylabel("Shape-Weighted Accuracy")

        lines, labels = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + l2, labels + lab2, loc="upper right")

        plt.title(f"{run_name}: Loss & Accuracy Curves\nDataset: SPR_BENCH")
        fname = os.path.join(working_dir, f"{run_name}_loss_swa_SPR_BENCH.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating curve plot for {run_name}: {e}")
        plt.close()

# -------------------- aggregated test metrics --------------------
try:
    names, test_loss_vals, test_swa_vals = [], [], []
    for run_name, run_data in runs.items():
        names.append(run_name)
        test_loss_vals.append(run_data.get("test", {}).get("loss", np.nan))
        test_swa_vals.append(run_data.get("test", {}).get("SWA", np.nan))

    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, test_loss_vals, width=width, label="Test Loss")
    plt.bar(x + width / 2, test_swa_vals, width=width, label="Test SWA")
    plt.ylabel("Value")
    plt.xticks(x, names, rotation=15)
    plt.ylim(0, max(max(test_loss_vals), max(test_swa_vals)) * 1.1)
    plt.title("Test Performance Across Model Dimensions\nDataset: SPR_BENCH")
    plt.legend()
    fname = os.path.join(working_dir, "test_metrics_summary_SPR_BENCH.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated metrics plot: {e}")
    plt.close()

# -------------------- print evaluation metrics --------------------
print("Final Test Metrics:")
for run_name, run_data in runs.items():
    m = run_data.get("test", {})
    print(
        f"{run_name}: Loss={m.get('loss', np.nan):.4f}, SWA={m.get('SWA', np.nan):.4f}"
    )
