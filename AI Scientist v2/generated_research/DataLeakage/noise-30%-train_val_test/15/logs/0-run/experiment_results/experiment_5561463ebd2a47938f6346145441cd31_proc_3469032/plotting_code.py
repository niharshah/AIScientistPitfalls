import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
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

# ---------- per-dataset plots ----------
test_scores = {}
for dname, d in experiment_data.items():
    epochs = d.get("epochs", [])
    tr_loss = d["losses"].get("train", [])
    val_loss = d["losses"].get("val", [])
    tr_f1 = d["metrics"].get("train", [])
    val_f1 = d["metrics"].get("val", [])
    preds = d.get("predictions", [])
    gts = d.get("ground_truth", [])
    test_f1 = d.get("test_macroF1", None)
    if test_f1 is not None:
        test_scores[dname] = test_f1
        print(f"{dname} – Test Macro-F1: {test_f1:.4f}")

    # ---- loss curve ----
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname}: Train vs Val Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dname}: {e}")
        plt.close()

    # ---- Macro-F1 curve ----
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dname}: Train vs Val Macro-F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_macroF1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Macro-F1 curve for {dname}: {e}")
        plt.close()

    # ---- confusion matrix ----
    try:
        if preds and gts:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(gts, preds, normalize="true")
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dname}: Normalized Confusion Matrix\nLeft: GT, Right: Pred")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

    # ---- class distribution ----
    try:
        if preds and gts:
            unique = sorted(set(gts + preds))
            gt_cnt = [gts.count(u) for u in unique]
            pr_cnt = [preds.count(u) for u in unique]
            x = np.arange(len(unique))
            plt.figure()
            plt.bar(x - 0.2, gt_cnt, width=0.4, label="Ground Truth")
            plt.bar(x + 0.2, pr_cnt, width=0.4, label="Predictions")
            plt.xlabel("Class")
            plt.ylabel("# Samples")
            plt.title(f"{dname}: Class Distribution")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dname}_class_distribution.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating class distribution for {dname}: {e}")
        plt.close()

# ---------- cross-dataset comparison ----------
if len(test_scores) > 1:
    try:
        names, scores = zip(*test_scores.items())
        plt.figure()
        plt.bar(names, scores)
        plt.ylabel("Test Macro-F1")
        plt.ylim(0, 1)
        plt.title("Comparison of Test Macro-F1 Across Datasets")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "datasets_test_macroF1_comparison.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        plt.close()
