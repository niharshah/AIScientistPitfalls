import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- iterate datasets ----------
for dset, ddata in experiment_data.items():
    # basic containers
    losses = ddata.get("losses", {})
    metrics = ddata.get("metrics", {})
    epochs = list(range(1, len(losses.get("train", [])) + 1))

    # ---- 1. Loss curve ----
    try:
        plt.figure()
        plt.plot(epochs, losses.get("train", []), label="train")
        plt.plot(epochs, losses.get("val", []), linestyle="--", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset}: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_curve.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting loss for {dset}: {e}")
        plt.close()

    # ---- 2. Metric curves (MCC, F1, etc.) ----
    try:
        for key in metrics.keys():
            if key.startswith("train_"):
                metric_name = key[len("train_") :]
                train_vals = metrics[key]
                val_vals = metrics.get(f"val_{metric_name}", [])
                plt.figure()
                plt.plot(epochs, train_vals, label="train")
                plt.plot(epochs, val_vals, linestyle="--", label="val")
                plt.xlabel("Epoch")
                plt.ylabel(metric_name.upper())
                plt.title(f"{dset}: Training vs Validation {metric_name.upper()}")
                plt.legend()
                fname = os.path.join(working_dir, f"{dset}_{metric_name}_curve.png")
                plt.savefig(fname, dpi=150)
                plt.close()
    except Exception as e:
        print(f"Error plotting metrics for {dset}: {e}")
        plt.close()

    # ---- 3. Confusion matrix ----
    try:
        preds = np.array(ddata.get("predictions", []))
        gts = np.array(ddata.get("ground_truth", []))
        if preds.size and gts.size:
            cm = np.zeros((2, 2), dtype=int)
            for g, p in zip(gts, preds):
                cm[g, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.title(f"{dset} Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.colorbar()
            fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
            plt.savefig(fname, dpi=150)
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dset}: {e}")
        plt.close()

    # ---- 4. Quick numeric report ----
    try:
        val_mcc = metrics.get("val_MCC", [None])[-1]
        test_mcc = np.nan if "mcc" not in ddata else ddata["mcc"]
        print(f"{dset} | Final Val MCC: {val_mcc} | Test MCC: {test_mcc}")
    except Exception:
        pass
