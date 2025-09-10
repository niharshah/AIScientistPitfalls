import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- container for comparison plot ----------
val_curves = {}

# ---------- per-dataset visualisation ----------
for dset, blob in experiment_data.items():
    print(f"{dset}  Test Macro-F1: {blob.get('test_macroF1', 'N/A'):.4f}")
    epochs = blob.get("epochs", [])
    tr_loss = blob.get("losses", {}).get("train", [])
    val_loss = blob.get("losses", {}).get("val", [])
    tr_f1 = blob.get("metrics", {}).get("train", [])
    val_f1 = blob.get("metrics", {}).get("val", [])
    preds = blob.get("predictions", [])
    gts = blob.get("ground_truth", [])

    # store for later comparison plot
    if epochs and val_f1:
        val_curves[dset] = (epochs, val_f1)

    # ---------- loss curve ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset}: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dset}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dset}: {e}")
        plt.close()

    # ---------- macro-F1 curve ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dset}: Training vs Validation Macro-F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dset}_macroF1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating macro-F1 curve for {dset}: {e}")
        plt.close()

    # ---------- confusion matrix ----------
    try:
        from sklearn.metrics import confusion_matrix

        if preds and gts:
            cm = confusion_matrix(gts, preds, normalize="true")
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(
                f"{dset}: Normalized Confusion Matrix\n"
                "Left: Ground Truth, Right: Predictions"
            )
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

# ---------- comparison plot across datasets ----------
if len(val_curves) > 1:
    try:
        plt.figure()
        for dset, (ep, vf1) in val_curves.items():
            plt.plot(ep, vf1, label=dset)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Macro-F1")
        plt.title("Validation Macro-F1 Comparison Across Datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "all_datasets_val_macroF1.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        plt.close()
