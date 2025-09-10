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

# ---------- per-dataset plots ----------
test_scores = {}
for dname, dct in experiment_data.items():
    try:
        epochs = dct["epochs"]
        tr_loss = dct["losses"]["train"]
        val_loss = dct["losses"]["val"]
        tr_f1 = dct["metrics"]["train"]
        val_f1 = dct["metrics"]["val"]
        preds = dct.get("predictions", [])
        gts = dct.get("ground_truth", [])
        test_macro = dct.get("test_macroF1", None)
        if test_macro is not None:
            test_scores[dname] = test_macro
            print(f"{dname} ‑ Test Macro-F1: {test_macro:.4f}")
    except Exception as e:
        print(f"Dataset {dname} missing keys: {e}")
        continue

    # ---- 1. Loss curve ----
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname}: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dname}: {e}")
        plt.close()

    # ---- 2. Macro-F1 curve ----
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dname}: Training vs Validation Macro-F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_macroF1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Macro-F1 curve for {dname}: {e}")
        plt.close()

    # ---- 3. Confusion matrix ----
    try:
        if preds and gts:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(gts, preds, normalize="true")
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(
                f"{dname}: Normalized Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
            )
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

# ---------- comparison bar chart ----------
try:
    if test_scores:
        plt.figure()
        names = list(test_scores.keys())
        scores = [test_scores[n] for n in names]
        plt.bar(names, scores, color="skyblue")
        plt.ylabel("Test Macro-F1")
        plt.title("Dataset Comparison: Test Macro-F1")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "comparison_test_macroF1.png"))
        plt.close()
except Exception as e:
    print(f"Error creating comparison plot: {e}")
    plt.close()
