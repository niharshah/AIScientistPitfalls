import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------ load experiment data ----------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

loss_train = data.get("losses", {}).get("train", [])
loss_val = data.get("losses", {}).get("val", [])
val_swa = data.get("metrics", {}).get("val", [])
test_swa = data.get("metrics", {}).get("test", None)
preds = data.get("predictions", [])
gts = data.get("ground_truth", [])

epochs = np.arange(1, len(loss_train) + 1)

# ------------------------ 1. Loss curves ----------------------------------
try:
    plt.figure(figsize=(6, 4))
    if loss_train:
        plt.plot(epochs, loss_train, label="Train")
    if loss_val:
        plt.plot(epochs, loss_val, linestyle="--", label="Validation")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------------ 2. Validation SWA -------------------------------
try:
    if val_swa:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs[: len(val_swa)], val_swa, marker="o")
        plt.title("SPR_BENCH Validation Shape-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_SWA_curve.png"))
        plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# ------------------------ 3. Confusion matrix -----------------------------
try:
    if preds and gts and len(preds) == len(gts):
        classes = sorted(set(gts + preds))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for gt, pr in zip(gts, preds):
            cm[classes.index(gt), classes.index(pr)] += 1
        plt.figure(figsize=(4, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046)
        plt.title("SPR_BENCH Confusion Matrix\nRows: GT, Cols: Pred")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------------ print final metric ------------------------------
if test_swa is not None:
    print(f"Final Test SWA: {test_swa:.4f}")
