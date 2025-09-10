import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths & load ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- plotting ----------
ds_key = "BinaryNgramFeature"
for mode, run in experiment_data.get(ds_key, {}).items():
    metrics = run.get("metrics", {})
    train_acc = np.asarray(metrics.get("train_acc", []), dtype=float)
    val_acc = np.asarray(metrics.get("val_acc", []), dtype=float)
    train_loss = np.asarray(run.get("losses", {}).get("train", []), dtype=float)
    val_loss = np.asarray(metrics.get("val_loss", []), dtype=float)
    steps = np.arange(1, len(train_acc) + 1)

    # 1) accuracy plot --------------------------------------------------------
    try:
        plt.figure()
        plt.plot(steps, train_acc, label="Train Accuracy")
        plt.plot(steps, val_acc, label="Val Accuracy")
        plt.title(f"{ds_key} ({mode}) – Accuracy Curves")
        plt.xlabel("Training Step")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_{mode}_acc_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {mode}: {e}")
        plt.close()

    # 2) loss plot ------------------------------------------------------------
    try:
        plt.figure()
        if train_loss.size:
            plt.plot(steps, train_loss, label="Train Loss")
        if val_loss.size:
            plt.plot(steps, val_loss, label="Val Loss")
        plt.title(f"{ds_key} ({mode}) – Loss Curves")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_{mode}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {mode}: {e}")
        plt.close()

# ---------- final metric printout ----------
if ds_key in experiment_data:
    print("\n=== Final Test Metrics ===")
    for mode, run in experiment_data[ds_key].items():
        print(
            f"{mode:6s} | test_acc = {run.get('test_acc'):.4f} | "
            f"test_rfs = {run.get('test_rfs'):.4f}"
        )
