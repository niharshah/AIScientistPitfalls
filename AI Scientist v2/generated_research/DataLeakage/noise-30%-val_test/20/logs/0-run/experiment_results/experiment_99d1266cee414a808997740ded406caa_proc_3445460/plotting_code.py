import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

# ------------------------------------------------------------------
# Helper: get epochs
epochs = np.arange(1, len(exp.get("losses", {}).get("train", [])) + 1)

# 1) Loss curves ----------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, exp["losses"]["train"], label="Train", color="tab:blue")
    plt.plot(epochs, exp["losses"]["val"], label="Validation", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) Macro-F1 curve -------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, exp["metrics"]["val_macroF1"], marker="o", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Validation Macro-F1 over Epochs")
    fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Macro-F1 curve: {e}")
    plt.close()

# 3) Complexity-Weighted Accuracy curve -----------------------------
try:
    plt.figure()
    plt.plot(epochs, exp["metrics"]["val_CWA"], marker="s", color="tab:red")
    plt.xlabel("Epoch")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.title("SPR_BENCH: Validation CWA over Epochs")
    fname = os.path.join(working_dir, "SPR_BENCH_CWA_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CWA curve: {e}")
    plt.close()

# 4) Confusion matrix (final epoch) ---------------------------------
try:
    preds = np.array(exp["predictions"])
    gts = np.array(exp["ground_truth"])
    cm = np.zeros((2, 2), dtype=int)
    for p, t in zip(preds, gts):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Generated Predictions"
    )
    plt.colorbar()
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print final evaluation metrics
if exp:
    final_F1 = exp["metrics"]["val_macroF1"][-1]
    final_CWA = exp["metrics"]["val_CWA"][-1]
    print(f"Final Validation Macro-F1: {final_F1:.4f}")
    print(f"Final Validation CWA     : {final_CWA:.4f}")
