import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ------------------- setup & load data -----------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------- per-dataset plots -----------------------------------
val_curves = {}
for name, rec in experiment_data.items():
    try:
        epochs = rec["epochs"]
        tr_loss = rec["losses"]["train"]
        val_loss = rec["losses"]["val"]
        tr_f1 = rec["metrics"]["train"]
        val_f1 = rec["metrics"]["val"]
        preds = rec.get("predictions", [])
        gts = rec.get("ground_truth", [])
        test_f1 = rec.get("test_macroF1", None)

        if test_f1 is not None:
            print(f"{name} – Test Macro-F1: {test_f1:.4f}")

        # 1) Loss curve
        try:
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{name}: Training vs Validation Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{name}_loss_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error plotting loss for {name}: {e}")
            plt.close()

        # 2) Macro-F1 curve
        try:
            plt.figure()
            plt.plot(epochs, tr_f1, label="Train Macro-F1")
            plt.plot(epochs, val_f1, label="Validation Macro-F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{name}: Training vs Validation Macro-F1")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{name}_macroF1_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error plotting f1 for {name}: {e}")
            plt.close()

        # 3) Confusion matrix
        try:
            if preds and gts:
                cm = confusion_matrix(gts, preds, normalize="true")
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title(
                    f"{name}: Normalized Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
                )
                plt.tight_layout()
                plt.savefig(os.path.join(working_dir, f"{name}_confusion_matrix.png"))
                plt.close()
        except Exception as e:
            print(f"Error plotting confusion matrix for {name}: {e}")
            plt.close()

        val_curves[name] = (epochs, val_f1)
    except Exception as e:
        print(f"Error processing dataset {name}: {e}")

# ------------------- multi-dataset comparison ----------------------------
if len(val_curves) >= 2:
    try:
        plt.figure()
        for ds_name, (ep, f1s) in val_curves.items():
            plt.plot(ep, f1s, label=f"{ds_name} Val Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("Validation Macro-F1 Comparison Across Datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "comparison_val_macroF1.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting comparison curve: {e}")
        plt.close()
