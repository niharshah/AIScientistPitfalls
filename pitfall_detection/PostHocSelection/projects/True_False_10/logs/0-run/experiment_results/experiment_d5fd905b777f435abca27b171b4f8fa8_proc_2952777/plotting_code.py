import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ----------------------------------------------------------------------
# Helper to fetch the single (model, dataset) entry
def _get_entry(edict):
    for model_k, model_v in edict.items():
        for dset_k, dset_v in model_v.items():
            return model_k, dset_k, dset_v
    return None, None, None


model_name, dset_name, ed = _get_entry(experiment_data)

if ed is None:
    print("No experiment data found, nothing to plot.")
else:
    epochs = np.array(ed["epochs"])
    train_loss = np.array(ed["losses"]["train"])
    val_loss = np.array(ed["losses"]["val"])
    val_swa = np.array(ed["metrics"]["val"])
    # guard against None values that may appear in metrics
    val_swa = np.where(val_swa == None, np.nan, val_swa.astype(float))

    # ---------------- Plot 1: Loss curves ------------------------------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset_name}: Train vs Val Loss ({model_name})")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dset_name}_{model_name}_loss_curves.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------------- Plot 2: Validation SWA ---------------------------
    try:
        plt.figure()
        plt.plot(epochs, val_swa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title(f"{dset_name}: Validation SWA ({model_name})")
        plt.savefig(os.path.join(working_dir, f"{dset_name}_{model_name}_val_SWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # ---------------- Plot 3: Final epoch prediction histogram ---------
    try:
        final_gt = ed["ground_truth"][-1]
        final_pred = ed["predictions"][-1]
        labels = sorted(set(final_gt) | set(final_pred))
        gt_counts = [final_gt.count(l) for l in labels]
        pr_counts = [final_pred.count(l) for l in labels]

        x = np.arange(len(labels))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pr_counts, width, label="Predicted")
        plt.xlabel("Label ID")
        plt.ylabel("Count")
        plt.title(f"{dset_name}: Final Epoch Label Distribution ({model_name})")
        plt.xticks(x, labels)
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dset_name}_{model_name}_final_epoch_hist.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating histogram plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Print summary metrics
    try:
        best_idx = np.nanargmin(val_loss)
        best_val_loss = val_loss[best_idx]
        best_swa = val_swa[best_idx]
        print(f"Best Validation Loss: {best_val_loss:.4f} at epoch {epochs[best_idx]}")
        print(f"Corresponding Validation SWA: {best_swa:.4f}")
    except Exception as e:
        print(f"Error computing summary metrics: {e}")
