import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- per-dataset plots ----------
best_vals = {}
for dset, info in experiment_data.items():
    # ----- loss curves -----
    try:
        plt.figure()
        epochs = np.arange(1, len(info["losses"]["train"]) + 1)
        plt.plot(epochs, info["losses"]["train"], label="train")
        plt.plot(epochs, info["losses"]["val"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset} – Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
        plt.close()

    # ----- metric curves (assume first key is primary val metric) -----
    try:
        metric_name, metric_vals = next(iter(info["metrics"].items()))
        plt.figure()
        plt.plot(np.arange(1, len(metric_vals) + 1), metric_vals, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{dset} – Validation {metric_name} per Epoch")
        fname = os.path.join(working_dir, f"{dset}_{metric_name}_curve.png")
        plt.savefig(fname)
        plt.close()
        best_vals[dset] = max(metric_vals)
    except Exception as e:
        print(f"Error creating metric plot for {dset}: {e}")
        plt.close()

    # ----- confusion matrix -----
    try:
        preds = np.array(info.get("predictions", []))
        gts = np.array(info.get("ground_truth", []))
        if preds.size and gts.size:
            n_cls = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dset} – Confusion Matrix\n(rows = GT, cols = Pred)")
            fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

# ---------- cross-dataset comparison ----------
try:
    if len(best_vals) > 1:
        plt.figure()
        names, vals = zip(*best_vals.items())
        plt.bar(names, vals, color="skyblue")
        plt.ylabel("Best Validation Metric")
        plt.title("Best Validation Metric Across Datasets")
        fname = os.path.join(working_dir, "datasets_best_val_metric.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating comparison plot: {e}")
    plt.close()

# ---------- print quick summary ----------
for d, v in best_vals.items():
    print(f"{d}: best validation metric = {v:.4f}")
