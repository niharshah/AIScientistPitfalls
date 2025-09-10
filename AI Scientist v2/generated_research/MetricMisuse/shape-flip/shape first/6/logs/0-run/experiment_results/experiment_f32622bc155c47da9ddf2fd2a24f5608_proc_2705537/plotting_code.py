import matplotlib.pyplot as plt
import numpy as np
import os

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

exp_key = "NoPositionalEmbedding"
dset_key = "SPR_BENCH"
rec = experiment_data.get(exp_key, {}).get(dset_key, {})


# ---------- helper to close safely ----------
def safe_close():
    if plt.get_fignums():
        plt.close()


# ---------- 1. Loss curve ----------
try:
    train_loss = rec.get("losses", {}).get("train", [])
    val_loss = rec.get("losses", {}).get("val", [])
    if train_loss and val_loss:
        plt.figure(figsize=(6, 4))
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Val Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    safe_close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    safe_close()

# ---------- 2. Validation SWA curve ----------
try:
    val_swa = rec.get("SWA", {}).get("val", [])
    if val_swa:
        plt.figure(figsize=(6, 4))
        plt.plot(val_swa, marker="o")
        plt.title("SPR_BENCH: Validation Shape-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_swa_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    safe_close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    safe_close()

# ---------- 3. Confusion matrix ----------
try:
    preds = rec.get("predictions", [])
    truths = rec.get("ground_truth", [])
    if preds and truths:
        labels = sorted(set(truths + preds))
        n_cls = len(labels)
        conf = np.zeros((n_cls, n_cls), dtype=int)
        idx = {l: i for i, l in enumerate(labels)}
        for t, p in zip(truths, preds):
            conf[idx[t], idx[p]] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(conf, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(range(n_cls), labels, rotation=90)
        plt.yticks(range(n_cls), labels)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    safe_close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    safe_close()
