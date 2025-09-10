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

runs = experiment_data.get("num_epochs", {})
# -------------------- 1-4: loss curves per run --------------------
for run_idx, (run_name, run_data) in enumerate(runs.items()):
    if run_idx >= 4:  # safety: plot at most 4 loss figures
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
    except Exception as e:
        print(f"Error creating loss plot for {run_name}: {e}")
        plt.close()

# -------------------- 5: aggregated test metrics --------------------
try:
    labels, crwa_vals, swa_vals, cwa_vals = [], [], [], []
    for run_name, run_data in runs.items():
        labels.append(run_name)
        test_metrics = run_data.get("metrics", {}).get("test", {})
        crwa_vals.append(test_metrics.get("CRWA", 0))
        swa_vals.append(test_metrics.get("SWA", 0))
        cwa_vals.append(test_metrics.get("CWA", 0))

    x = np.arange(len(labels))
    w = 0.25
    plt.figure(figsize=(8, 4))
    plt.bar(x - w, crwa_vals, width=w, label="CRWA")
    plt.bar(x, swa_vals, width=w, label="SWA")
    plt.bar(x + w, cwa_vals, width=w, label="CWA")
    plt.xticks(x, labels, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Test Metrics Across Epoch Settings\nDataset: SPR_BENCH (toy)")
    plt.legend()
    fname = os.path.join(working_dir, "test_metrics_comparison_SPR_BENCH.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated metrics plot: {e}")
    plt.close()

# -------------------- print evaluation metrics --------------------
print("Final Test Metrics:")
for run_name, run_data in runs.items():
    m = run_data.get("metrics", {}).get("test", {})
    print(
        f"{run_name}: CRWA={m.get('CRWA',0):.4f}, SWA={m.get('SWA',0):.4f}, CWA={m.get('CWA',0):.4f}"
    )
