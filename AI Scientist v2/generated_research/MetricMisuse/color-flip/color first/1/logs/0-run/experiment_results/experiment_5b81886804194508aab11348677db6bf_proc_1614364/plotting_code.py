import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ paths & load
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

tag, dname = "frozen_embeddings", "SPR"
exp = experiment_data.get(tag, {}).get(dname, {})

loss_tr = exp.get("losses", {}).get("train", [])
loss_val = exp.get("losses", {}).get("val", [])
metrics_val = exp.get("metrics", {}).get("val", [])
test_metrics = exp.get("metrics", {}).get("test", {})
epochs = list(range(1, len(loss_tr) + 1))

# ------------------------------------------------------------------ PLOT 1: loss curves
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Loss Curves\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------ PLOT 2: accuracy curves
try:
    cwa = [m["cwa"] for m in metrics_val]
    swa = [m["swa"] for m in metrics_val]
    cva = [m["cva"] for m in metrics_val]
    plt.figure()
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, cva, label="CVA")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR Weighted Accuracies over Epochs\nLeft: CWA, Middle: SWA, Right: CVA")
    plt.legend()
    fname = os.path.join(working_dir, f"{dname}_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------------ PLOT 3: final test metrics
try:
    labels = ["CWA", "SWA", "CVA"]
    vals = [test_metrics.get(k.lower(), 0) for k in labels]
    plt.figure()
    plt.bar(labels, vals)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("SPR Test Metrics\nBar chart of final weighted accuracies")
    fname = os.path.join(working_dir, f"{dname}_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metric plot: {e}")
    plt.close()

# ------------------------------------------------------------------ print metrics
if test_metrics:
    print("Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k.upper()}: {v:.4f}")
