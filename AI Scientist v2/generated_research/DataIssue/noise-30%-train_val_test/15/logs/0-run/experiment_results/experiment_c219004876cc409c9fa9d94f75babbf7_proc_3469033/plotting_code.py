import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------- setup and data loading ------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ----------------------------- plotting --------------------------------
for dset_name, d in experiment_data.items():
    try:
        print(f"{dset_name}  Test Macro-F1: {d.get('test_macroF1', np.nan):.4f}")
    except Exception as e:
        print(f"Could not read test metric for {dset_name}: {e}")

    epochs = d.get("epochs", [])
    tr_loss = d.get("losses", {}).get("train", [])
    val_loss = d.get("losses", {}).get("val", [])
    tr_f1 = d.get("metrics", {}).get("train", [])
    val_f1 = d.get("metrics", {}).get("val", [])
    preds = d.get("predictions", [])
    gts = d.get("ground_truth", [])

    # 1) Loss curve ------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset_name}: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dset_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"{dset_name} loss curve error: {e}")
        plt.close()

    # 2) Macro-F1 curve --------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dset_name}: Training vs Validation Macro-F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dset_name}_macroF1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"{dset_name} Macro-F1 curve error: {e}")
        plt.close()

    # 3) Confusion matrix ------------------------------------------------
    if preds and gts:
        try:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(gts, preds, normalize="true")
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(
                f"{dset_name}: Normalized Confusion Matrix\n"
                "Left: Ground Truth, Right: Predictions"
            )
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset_name}_confusion_matrix.png"))
            plt.close()
        except Exception as e:
            print(f"{dset_name} confusion matrix error: {e}")
            plt.close()

    # 4) Class distribution bar chart -----------------------------------
    if preds and gts:
        try:
            from collections import Counter

            true_cnt = Counter(gts)
            pred_cnt = Counter(preds)
            labels = sorted(set(list(true_cnt.keys()) + list(pred_cnt.keys())))
            true_vals = [true_cnt.get(l, 0) for l in labels]
            pred_vals = [pred_cnt.get(l, 0) for l in labels]

            x = np.arange(len(labels))
            width = 0.35
            plt.figure()
            plt.bar(x - width / 2, true_vals, width, label="Ground Truth")
            plt.bar(x + width / 2, pred_vals, width, label="Predicted")
            plt.xlabel("Class Label")
            plt.ylabel("Count")
            plt.title(
                f"{dset_name}: Class Distribution\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.xticks(x, labels)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{dset_name}_class_distribution.png")
            )
            plt.close()
        except Exception as e:
            print(f"{dset_name} class distribution error: {e}")
            plt.close()
