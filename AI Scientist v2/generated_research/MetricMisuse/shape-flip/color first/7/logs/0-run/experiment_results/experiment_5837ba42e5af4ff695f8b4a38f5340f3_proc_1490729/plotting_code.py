import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment results ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_key = "SPR_BENCH"
runs = experiment_data.get("batch_size_tuning", {}).get(ds_key, {})
batch_sizes = sorted(runs.keys(), key=int)[:5]  # keep order & cap to 5 plots total

# --------- per-batch-size learning curves ---------
for bs in batch_sizes[:-1]:  # leave 1 slot for summary so total <=5
    log = runs[bs]
    epochs = range(1, len(log["losses"]["train"]) + 1)
    try:
        plt.figure()
        # loss curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, log["losses"]["train"], label="Train")
        plt.plot(epochs, log["losses"]["val"], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss (bs={bs})")
        plt.legend()
        # accuracy curves
        tr_acc = [m["acc"] for m in log["metrics"]["train"]]
        vl_acc = [m["acc"] for m in log["metrics"]["val"]]
        plt.subplot(1, 2, 2)
        plt.plot(epochs, tr_acc, label="Train")
        plt.plot(epochs, vl_acc, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy (bs={bs})")
        plt.legend()
        plt.suptitle(f"{ds_key} – Training/Validation Curves (batch={bs})")
        fname = os.path.join(working_dir, f"{ds_key}_bs{bs}_train_val_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating curves for batch size {bs}: {e}")
        plt.close()

# --------- summary plot across batch sizes ---------
try:
    plt.figure()
    test_acc = [runs[bs]["test"]["acc"] for bs in batch_sizes]
    test_cowa = [runs[bs]["test"]["cowa"] for bs in batch_sizes]
    x = np.arange(len(batch_sizes))
    width = 0.35
    plt.bar(x - width / 2, test_acc, width, label="Accuracy")
    plt.bar(x + width / 2, test_cowa, width, label="CoWA")
    plt.xticks(x, batch_sizes)
    plt.xlabel("Batch Size")
    plt.ylabel("Metric")
    plt.title(f"{ds_key} – Test Metrics vs. Batch Size")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_key}_test_metrics_summary.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating summary plot: {e}")
    plt.close()
