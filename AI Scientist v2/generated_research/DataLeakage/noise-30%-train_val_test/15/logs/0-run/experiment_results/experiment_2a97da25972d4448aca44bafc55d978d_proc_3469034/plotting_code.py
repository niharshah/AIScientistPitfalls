import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------ prepare paths ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load experiment log ------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------ iterate over datasets ----------
for ds_name, ds in experiment_data.items():
    test_f1 = ds.get("test_macroF1", None)
    if test_f1 is not None:
        print(f"{ds_name} – Test Macro-F1: {test_f1:.4f}")
    epochs = ds.get("epochs", [])
    tr_loss = ds.get("losses", {}).get("train", [])
    val_loss = ds.get("losses", {}).get("val", [])
    tr_f1 = ds.get("metrics", {}).get("train", [])
    val_f1 = ds.get("metrics", {}).get("val", [])
    preds = ds.get("predictions", [])
    gts = ds.get("ground_truth", [])

    # -------- loss curve ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name}: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {ds_name}: {e}")
        plt.close()

    # -------- macro-F1 curve ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{ds_name}: Training vs Validation Macro-F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_macroF1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Macro-F1 curve for {ds_name}: {e}")
        plt.close()

    # -------- confusion matrix ----------
    try:
        if preds and gts:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(gts, preds, normalize="true")
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(
                f"{ds_name}: Normalized Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()
