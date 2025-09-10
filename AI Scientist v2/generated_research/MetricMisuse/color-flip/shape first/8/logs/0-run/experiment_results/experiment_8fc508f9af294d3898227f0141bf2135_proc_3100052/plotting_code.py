import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def safe_get(d, *keys, default=list()):
    for k in keys:
        d = d.get(k, {})
    return d if isinstance(d, list) else default


# --------------- load experiment data ---------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------- plotting per dataset --------------------------
for dset_name, logs in experiment_data.items():
    losses_tr = safe_get(logs, "losses", "train")
    losses_val = safe_get(logs, "losses", "val")
    ccwa_val = safe_get(logs, "metrics", "val_CCWA")
    preds_list = logs.get("predictions", [])
    gt_list = logs.get("ground_truth", [])
    stamps = safe_get(logs, "timestamps")

    epochs = list(range(1, len(losses_tr) + 1))

    # 1) Train vs Val loss
    try:
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train")
        plt.plot(epochs, losses_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset_name}: Train vs Val Loss")
        plt.legend()
        fname = f"{dset_name}_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Loss curve error ({dset_name}): {e}")
        plt.close()

    # 2) Validation CCWA
    try:
        plt.figure()
        plt.plot(epochs, ccwa_val, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("CCWA")
        plt.title(f"{dset_name}: Validation CCWA")
        fname = f"{dset_name}_val_CCWA.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"CCWA plot error ({dset_name}): {e}")
        plt.close()

    # 3) Val Loss vs CCWA scatter
    try:
        if len(losses_val) == len(ccwa_val) > 0:
            plt.figure()
            plt.scatter(losses_val, ccwa_val, c=epochs, cmap="viridis")
            plt.colorbar(label="Epoch")
            plt.xlabel("Validation Loss")
            plt.ylabel("CCWA")
            plt.title(f"{dset_name}: Loss vs CCWA")
            fname = f"{dset_name}_loss_vs_CCWA.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Scatter plot error ({dset_name}): {e}")
        plt.close()

    # 4) Timestamped validation loss
    try:
        if len(stamps) == len(losses_val) > 0:
            times = range(len(stamps))  # simple index for equal spacing
            plt.figure()
            plt.plot(times, losses_val, marker="s")
            plt.xlabel("Checkpoint")
            plt.ylabel("Val Loss")
            plt.title(f"{dset_name}: Validation Loss over Time")
            fname = f"{dset_name}_val_loss_time.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Time plot error ({dset_name}): {e}")
        plt.close()

    # 5) Confusion matrix of final epoch
    try:
        if preds_list and gt_list:
            preds, gts = preds_list[-1], gt_list[-1]
            n_cls = max(max(preds), max(gts)) + 1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for p, t in zip(preds, gts):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dset_name}: Confusion Matrix (Final Epoch)")
            for i in range(n_cls):
                for j in range(n_cls):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = f"{dset_name}_confusion_matrix.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Confusion matrix error ({dset_name}): {e}")
        plt.close()

    # -------- optional console summary -------------------------
    if ccwa_val:
        print(f"{dset_name} final CCWA: {ccwa_val[-1]:.4f}")

print("Plotting complete – figures saved in", working_dir)
