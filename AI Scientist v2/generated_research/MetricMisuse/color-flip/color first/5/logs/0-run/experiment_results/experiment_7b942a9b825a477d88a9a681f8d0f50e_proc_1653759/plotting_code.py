import matplotlib.pyplot as plt
import numpy as np
import os

# --- setup ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

datasets = list(experiment_data.keys())
test_accs = {}

# ---------- per-dataset visualisations ----------
for dset in datasets:
    data = experiment_data.get(dset, {})
    losses = data.get("losses", {})
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    val_metrics = data.get("metrics", {}).get("val", [])
    test_metrics = data.get("metrics", {}).get("test", {})

    # ---- 1. loss curves ----
    try:
        if train_loss and val_loss:
            plt.figure(figsize=(6, 4))
            epochs = np.arange(1, len(train_loss) + 1)
            plt.plot(epochs, train_loss, label="train")
            plt.plot(epochs, val_loss, linestyle="--", label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset} — Train vs Val Loss\n(Left: train, Right: val)")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_loss_curves.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
        plt.close()

    # ---- 2. validation accuracy ----
    try:
        if val_metrics:
            accs = [m.get("acc", np.nan) for m in val_metrics]
            epochs = np.arange(1, len(accs) + 1)
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, accs, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dset} — Validation Accuracy across Epochs")
            fname = os.path.join(working_dir, f"{dset}_val_accuracy.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dset}: {e}")
        plt.close()

    # ---- 3. test metrics bar chart ----
    try:
        if test_metrics:
            metric_names = ["acc", "cwa", "swa", "ccwa"]
            values = [test_metrics.get(m, np.nan) for m in metric_names]
            plt.figure(figsize=(6, 4))
            plt.bar(metric_names, values, color="skyblue")
            plt.ylim(0, 1)
            plt.title(f"{dset} — Test Metrics")
            for i, v in enumerate(values):
                if not np.isnan(v):
                    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
            fname = os.path.join(working_dir, f"{dset}_test_metrics.png")
            plt.savefig(fname)
            test_accs[dset] = test_metrics.get("acc", np.nan)
        plt.close()
    except Exception as e:
        print(f"Error creating test metric plot for {dset}: {e}")
        plt.close()

    # ---- print metrics ----
    if test_metrics:
        print(f"\n{dset} TEST METRICS")
        for k, v in test_metrics.items():
            print(f"{k.upper():5s}: {v:.3f}")

# ---------- cross-dataset comparison ----------
try:
    if len(test_accs) > 1:
        plt.figure(figsize=(6, 4))
        names, vals = zip(*test_accs.items())
        plt.bar(names, vals, color="salmon")
        plt.ylim(0, 1)
        plt.title("Test Accuracy Comparison Across Datasets")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        fname = os.path.join(working_dir, "comparison_test_accuracy.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating comparison plot: {e}")
    plt.close()
