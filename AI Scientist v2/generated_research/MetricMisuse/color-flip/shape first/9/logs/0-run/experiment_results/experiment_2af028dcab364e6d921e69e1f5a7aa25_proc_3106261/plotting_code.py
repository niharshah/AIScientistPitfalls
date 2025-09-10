import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
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

ed = experiment_data.get("NoContrastivePretrain", {}).get("SPR_BENCH", {})
loss_tr = ed.get("losses", {}).get("train", [])
loss_val = ed.get("losses", {}).get("val", [])
comp_val = ed.get("metrics", {}).get("val", [])
preds = np.array(ed.get("predictions", []))
gts = np.array(ed.get("ground_truth", []))

# ---------- figure 1: loss ----------
try:
    if loss_tr and loss_val:
        plt.figure()
        epochs = np.arange(1, len(loss_tr) + 1)
        plt.plot(epochs, loss_tr, label="Train Loss")
        plt.plot(epochs, loss_val, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curve - NoContrastivePretrain")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_loss_curve_NoContrastivePretrain.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- figure 2: CompWA ----------
try:
    if comp_val:
        plt.figure()
        epochs = np.arange(1, len(comp_val) + 1)
        plt.plot(epochs, comp_val, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Comp-Weighted-Accuracy")
        plt.title("SPR_BENCH Validation CompWA - NoContrastivePretrain")
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_CompWA_NoContrastivePretrain.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating CompWA plot: {e}")
    plt.close()

# ---------- figure 3: confusion matrix ----------
try:
    if preds.size and gts.size:
        num_cls = len(np.unique(np.concatenate([preds, gts])))
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(
            "SPR_BENCH Confusion Matrix - NoContrastivePretrain\nLeft: Ground Truth, Right: Predictions"
        )
        for i in range(num_cls):
            for j in range(num_cls):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "SPR_BENCH_confusion_matrix_NoContrastivePretrain.png"
            )
        )
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- summary ----------
if comp_val:
    best_epoch = int(np.argmax(comp_val) + 1)
    best_score = comp_val[best_epoch - 1]
    print(f"Best Comp-Weighted-Accuracy: {best_score:.4f} at epoch {best_epoch}")
