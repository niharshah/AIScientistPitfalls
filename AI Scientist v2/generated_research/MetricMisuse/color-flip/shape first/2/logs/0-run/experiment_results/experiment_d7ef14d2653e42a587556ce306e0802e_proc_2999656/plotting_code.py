import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

jt = experiment_data.get("joint_training", {})
loss_tr = jt.get("losses", {}).get("train", [])
loss_val = jt.get("losses", {}).get("val", [])
metrics_val = jt.get("metrics", {}).get("val", [])
preds = jt.get("predictions", [])
gts = jt.get("ground_truth", [])


# ---------- helper ----------
def _safe_close():
    if plt.get_fignums():
        plt.close()


# ---------- FIG 1: loss curves ----------
try:
    if loss_tr and loss_val:
        epochs = range(1, len(loss_tr) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, loss_tr, label="Train")
        axes[1].plot(epochs, loss_val, label="Validation", color="orange")
        axes[0].set_title("Left: Train Loss (SPR_BENCH)")
        axes[1].set_title("Right: Validation Loss (SPR_BENCH)")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        fig.suptitle("SPR_BENCH Joint-Training Loss Curves")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_joint_training_loss_curves.png")
        )
    _safe_close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    _safe_close()

# ---------- FIG 2: metric curves ----------
try:
    if metrics_val:
        ep = [m["epoch"] for m in metrics_val]
        swa = [m["swa"] for m in metrics_val]
        cwa = [m["cwa"] for m in metrics_val]
        ccwa = [m["ccwa"] for m in metrics_val]
        plt.figure(figsize=(6, 4))
        plt.plot(ep, swa, label="SWA")
        plt.plot(ep, cwa, label="CWA")
        plt.plot(ep, ccwa, label="CCWA")
        plt.title("SPR_BENCH Validation Metrics Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_metric_curves.png"))
    _safe_close()
except Exception as e:
    print(f"Error creating metric curves: {e}")
    _safe_close()

# ---------- FIG 3: confusion matrix ----------
try:
    if preds and gts:
        import itertools

        classes = sorted(set(gts))
        n = len(classes)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(gts, preds):
            cm[t][p] += 1
        plt.figure(figsize=(4 + n / 2, 4 + n / 2))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("Confusion Matrix: SPR_BENCH\nLeft: Ground Truth, Right: Predictions")
        tick_marks = np.arange(n)
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        for i, j in itertools.product(range(n), range(n)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    _safe_close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    _safe_close()

# ---------- print best CCWA ----------
if metrics_val:
    best_ccwa = max(m["ccwa"] for m in metrics_val)
    print(f"Best validation CCWA: {best_ccwa:.4f}")
