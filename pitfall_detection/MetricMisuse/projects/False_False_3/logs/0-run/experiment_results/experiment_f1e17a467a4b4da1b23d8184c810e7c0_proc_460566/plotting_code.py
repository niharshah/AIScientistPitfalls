import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to safely fetch nested dicts
def get(d, *keys, default=None):
    for k in keys:
        if k not in d:
            return default
        d = d[k]
    return d


exp = get(experiment_data, "bag_of_embeddings", "SPR_BENCH", default={})
loss_tr = get(exp, "losses", "train", default=[])
loss_val = get(exp, "losses", "val", default=[])
swa_val = get(exp, "metrics", "val", default=[])
preds = np.array(get(exp, "predictions", default=[]))
golds = np.array(get(exp, "ground_truth", default=[]))

# ---------- Loss curve ----------
try:
    if len(loss_tr) and len(loss_val):
        epochs = np.arange(1, len(loss_tr) + 1)
        plt.figure()
        plt.plot(epochs, loss_tr, label="Train")
        plt.plot(epochs, loss_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
    else:
        print("Loss data missing; skipping loss curve.")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- Metric curve (SWA) ----------
try:
    if len(swa_val):
        epochs = np.arange(1, len(swa_val) + 1)
        plt.figure()
        plt.plot(epochs, swa_val, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH: Validation Shape-Weighted Accuracy")
        fname = os.path.join(working_dir, "SPR_BENCH_metric_curve.png")
        plt.savefig(fname)
    else:
        print("Metric data missing; skipping metric curve.")
    plt.close()
except Exception as e:
    print(f"Error creating metric curve: {e}")
    plt.close()

# ---------- Confusion matrix ----------
try:
    if preds.size and golds.size:
        labels = sorted(list(set(golds) | set(preds)))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for p, g in zip(preds, golds):
            cm[label_to_idx[g], label_to_idx[p]] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
    else:
        print("Prediction data missing; skipping confusion matrix.")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- Print evaluation metric ----------
if preds.size and golds.size:
    acc = (preds == golds).mean()
    print(f"Test Accuracy: {acc:.4f}")
