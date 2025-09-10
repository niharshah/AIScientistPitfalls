import matplotlib.pyplot as plt
import numpy as np
import os

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

# ---------- plotting ----------
for dset_name, d in experiment_data.items():
    epochs = d.get("epochs", [])
    tr_loss = d.get("losses", {}).get("train", [])
    val_loss = d.get("losses", {}).get("val", [])

    # 1) Loss curves ---------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, "--", label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset_name} – Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {dset_name}: {e}")
        plt.close()

    # 2) CmpWA curves --------------------------------------------------------
    try:
        tr_cmp = [m["CmpWA"] for m in d.get("metrics", {}).get("train", [])]
        val_cmp = [m["CmpWA"] for m in d.get("metrics", {}).get("val", [])]
        plt.figure()
        plt.plot(epochs, tr_cmp, label="Train")
        plt.plot(epochs, val_cmp, "--", label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title(f"{dset_name} – Training vs Validation CmpWA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dset_name}_CmpWA_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CmpWA curves for {dset_name}: {e}")
        plt.close()

    # 3) Confusion matrix (test set) ----------------------------------------
    try:
        ys = d.get("ground_truth", [])
        yp = d.get("predictions", [])
        if ys and yp:
            labels = sorted(set(ys + yp))
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(ys, yp):
                cm[t][p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(range(len(labels)), labels)
            plt.yticks(range(len(labels)), labels)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dset_name} – Confusion Matrix (Test Set)")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset_name}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset_name}: {e}")
        plt.close()

    # 4) Print test metrics --------------------------------------------------
    test = d.get("test", {})
    if test:
        print(
            f"{dset_name}: Test loss={test.get('loss', np.nan):.4f} | "
            f"CWA={test.get('CWA', np.nan):.3f} | "
            f"SWA={test.get('SWA', np.nan):.3f} | "
            f"CmpWA={test.get('CmpWA', np.nan):.3f}"
        )
