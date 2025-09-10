import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dset_name, dset_dict in experiment_data.items():
    losses_tr = dset_dict["losses"]["train"]
    losses_val = dset_dict["losses"]["val"]
    scwa_vals = dset_dict["metrics"]["val_SCWA"]
    preds = np.array(dset_dict.get("predictions", []))
    gts = np.array(dset_dict.get("ground_truth", []))
    epochs = np.arange(1, len(losses_tr) + 1)

    # ------------- 1. Loss curve ------------------
    try:
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train")
        plt.plot(epochs, losses_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset_name}: Loss Curve\nTraining vs Validation")
        plt.legend()
        save_name = os.path.join(working_dir, f"{dset_name}_loss_curve.png")
        plt.savefig(save_name)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dset_name}: {e}")
        plt.close()

    # ------------- 2. SCWA curve ------------------
    try:
        plt.figure()
        plt.plot(epochs, scwa_vals, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("SCWA")
        plt.ylim(0, 1)
        plt.title(f"{dset_name}: Validation SCWA Over Epochs")
        save_name = os.path.join(working_dir, f"{dset_name}_val_SCWA_curve.png")
        plt.savefig(save_name)
        plt.close()
    except Exception as e:
        print(f"Error creating SCWA curve for {dset_name}: {e}")
        plt.close()

    # ------------- 3. Confusion matrix ------------
    try:
        if preds.size and gts.size:
            n_cls = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dset_name}: Confusion Matrix (Validation Set)")
            plt.xticks(range(n_cls))
            plt.yticks(range(n_cls))
            save_name = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
            plt.savefig(save_name)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset_name}: {e}")
        plt.close()

    # ------------- Print metrics ------------------
    val_scwa_arr = np.array(scwa_vals, dtype=float)
    if val_scwa_arr.size:
        print(
            f"{dset_name} – Final Val SCWA: {val_scwa_arr[-1]:.4f} | "
            f"Best Val SCWA: {np.nanmax(val_scwa_arr):.4f}"
        )
