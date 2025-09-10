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

spr = experiment_data.get("SPR_BENCH", {})
loss_tr = spr.get("losses", {}).get("train", [])
loss_val = spr.get("losses", {}).get("val", [])
metrics_val = spr.get("metrics", {}).get("val", [])
preds = spr.get("predictions", [])
truth = spr.get("ground_truth", [])

epochs = range(1, len(loss_tr) + 1)

# ---------- FIG 1: Loss curves ----------
try:
    if loss_tr and loss_val:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, loss_tr, marker="o")
        axes[1].plot(epochs, loss_val, marker="o", color="orange")
        axes[0].set_title("Train Loss")
        axes[1].set_title("Validation Loss")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
        fig.suptitle("SPR_BENCH Loss Curves (Left: Train, Right: Validation)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- FIG 2: Metric curves ----------
try:
    if metrics_val:
        ccwa = [m["ccwa"] for m in metrics_val]
        swa = [m["swa"] for m in metrics_val]
        cwa = [m["cwa"] for m in metrics_val]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, ccwa, label="CCWA", marker="o")
        plt.plot(epochs, swa, label="SWA", linestyle="--")
        plt.plot(epochs, cwa, label="CWA", linestyle=":")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH Validation Metrics Across Epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric curves: {e}")
    plt.close()

# ---------- FIG 3: Confusion matrix ----------
try:
    if preds and truth:
        preds = np.array(preds)
        truth = np.array(truth)
        n_labels = int(max(preds.max(), truth.max())) + 1
        cm = np.zeros((n_labels, n_labels), dtype=int)
        for t, p in zip(truth, preds):
            cm[t, p] += 1
        plt.figure(figsize=(5, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, shrink=0.8)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix (GT vs Pred)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- quick metric print ----------
if metrics_val:
    best_ccwa = max(m["ccwa"] for m in metrics_val)
    print(f"Best CCWA achieved: {best_ccwa:.4f}")
