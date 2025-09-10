import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------
runs = {}
dataset_key = "SPR_BENCH"
try:
    runs = experiment_data["num_epochs_tuning"][dataset_key]
except Exception as e:
    print(f"Could not extract runs: {e}")

# ------------------------------------------------------------------
# Plot 1-3: loss curves per run
for run_name, run_data in list(runs.items())[:3]:  # should be epochs_10/20/30
    try:
        losses = run_data["losses"]
        tr = losses["train_loss"]
        vl = losses["val_loss"]
        ep = np.arange(1, len(tr) + 1)
        plt.figure()
        plt.plot(ep, tr, label="Train Loss")
        plt.plot(ep, vl, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset_key} – {run_name} | Training vs Validation Loss")
        plt.legend()
        fname = f"{dataset_key}_{run_name}_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {run_name}: {e}")
        plt.close()

# ------------------------------------------------------------------
# Plot 4: validation HWA curves for all runs together
try:
    plt.figure()
    for run_name, run_data in runs.items():
        hwa = run_data["losses"]["val_HWA"]
        ep = np.arange(1, len(hwa) + 1)
        plt.plot(ep, hwa, label=run_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation HWA")
    plt.title(f"{dataset_key} – Validation HWA Across Epoch Settings")
    plt.legend()
    fname = f"{dataset_key}_val_HWA_comparison.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating HWA comparison plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Plot 5: grouped bar chart of final test metrics
try:
    metrics = ["SWA", "CWA", "HWA"]
    run_names = list(runs.keys())
    n_runs = len(run_names)
    x = np.arange(len(metrics))
    width = 0.8 / n_runs
    plt.figure()
    for i, rn in enumerate(run_names):
        vals = [runs[rn]["metrics"]["test"][m] for m in metrics]
        plt.bar(x + i * width - width * (n_runs - 1) / 2, vals, width=width, label=rn)
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title(f"{dataset_key} – Test Metrics by Epoch Setting")
    plt.legend()
    fname = f"{dataset_key}_test_metrics_bar.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print final test metrics to console
for rn, rd in runs.items():
    print(f'{rn}: {rd["metrics"]["test"]}')
