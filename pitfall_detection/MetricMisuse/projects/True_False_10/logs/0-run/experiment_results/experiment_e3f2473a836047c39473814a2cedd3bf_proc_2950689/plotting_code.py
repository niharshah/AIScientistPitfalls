import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# helper ------------------------------------------------------------
def _count_shape(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [_count_shape(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# load experiment data ---------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# iterate over datasets --------------------------------------------
for ds_name, ds_data in experiment_data.items():
    epochs = np.arange(1, len(ds_data["losses"]["train"]) + 1)

    # -------------- plot 1: loss curves ---------------------------
    try:
        plt.figure()
        plt.plot(epochs, ds_data["losses"]["train"], label="Train")
        plt.plot(epochs, ds_data["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name}: Training vs Validation Loss\nDataset: {ds_name}")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # -------------- plot 2: validation SWA -----------------------
    try:
        val_swa = ds_data["metrics"]["val"]
        plt.figure()
        plt.plot(epochs, val_swa, marker="o", label="Validation SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title(f"{ds_name}: Validation SWA Across Epochs\nDataset: {ds_name}")
        plt.ylim(0, 1.05)
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_val_SWA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {ds_name}: {e}")
        plt.close()

    # -------------- evaluation printout --------------------------
    try:
        # best epoch based on stored SWA
        swa_vals = np.array(
            [v if v is not None else -1 for v in ds_data["metrics"]["val"]]
        )
        best_idx = int(np.argmax(swa_vals))
        preds = ds_data["predictions"][best_idx]
        gts = ds_data["ground_truth"][best_idx]
        # raw sequences were not stored for val set; reuse stored metric value
        best_swa = swa_vals[best_idx]
        # guard: if raw sequences available, recompute
        if "raw_seqs" in ds_data:
            best_swa = shape_weighted_accuracy(
                ds_data["raw_seqs"][best_idx], gts, preds
            )
        print(f"{ds_name}: best Validation SWA={best_swa:.4f} at epoch {best_idx+1}")
    except Exception as e:
        print(f"Error computing best SWA for {ds_name}: {e}")
