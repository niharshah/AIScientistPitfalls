import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- setup --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit(0)

spr = experiment_data.get("SPR_BENCH", {})
if not spr:
    print("No SPR_BENCH logs found.")
    exit(0)

epochs = spr["epochs"]
loss_tr = spr["losses"]["train"]
loss_val = spr["losses"]["val"]
swa_val = spr["metrics"]["val"]  # list of floats
swa_test = spr["metrics"]["test"]  # single float
gts = np.array(spr["ground_truth"])
preds = np.array(spr["predictions"])

# ------------------ Plot 1: loss curves -----------------
try:
    plt.figure()
    plt.plot(epochs, loss_tr, "--", label="train")
    plt.plot(epochs, loss_val, "-", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------ Plot 2: validation SWA --------------
try:
    plt.figure()
    plt.plot(epochs, swa_val, marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy (SWA)")
    plt.title("SPR_BENCH: Validation SWA Across Epochs")
    fname = os.path.join(working_dir, "SPR_BENCH_val_SWA_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve plot: {e}")
    plt.close()

# ------------------ Plot 3: confusion matrix ------------
try:
    if len(preds) and len(gts):
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xticks([0, 1], ["invalid", "valid"])
        plt.yticks([0, 1], ["invalid", "valid"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(
            "SPR_BENCH: Test Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
        )
        fname = os.path.join(working_dir, "SPR_BENCH_test_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ------------------ Plot 4: final test SWA --------------
try:
    plt.figure()
    plt.bar(["SWA"], [swa_test], color="steelblue")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Final Test SWA")
    fname = os.path.join(working_dir, "SPR_BENCH_test_SWA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test SWA bar plot: {e}")
    plt.close()

# ------------------ print metric ------------------------
print(f"Final test SWA: {swa_test:.4f}")
