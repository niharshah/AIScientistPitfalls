import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
data = experiment_data.get(dataset, {})

losses = data.get("losses", {})
metrics = data.get("metrics", {})
preds = np.array(data.get("predictions", []))
gts = np.array(data.get("ground_truth", []))
epochs = np.arange(1, len(losses.get("train", [])) + 1)

# ---------- 1) loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, [v for _, v in losses.get("train", [])], "--o", label="Train")
    plt.plot(epochs, [v for _, v in losses.get("val", [])], "-s", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{dataset} Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{dataset}_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- 2) metric curves ----------
try:
    plt.figure()
    styles = {"val_CWA": "-o", "val_SWA": "-s", "val_CpxWA": "-^"}
    for k, st in styles.items():
        vals = [v for _, v in metrics.get(k, [])]
        if vals:
            plt.plot(epochs, vals, st, label=k.replace("val_", ""))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset} Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{dataset}_metric_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric curves: {e}")
    plt.close()

# ---------- 3) final metric bar ----------
try:
    plt.figure()
    names, scores = [], []
    for k in ("val_CWA", "val_SWA", "val_CpxWA"):
        if metrics.get(k):
            names.append(k.replace("val_", ""))
            scores.append(metrics[k][-1][1])
    plt.bar(names, scores, color="skyblue")
    plt.ylabel("Final Accuracy")
    plt.title(f"{dataset} Final Validation Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{dataset}_final_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final metric bar: {e}")
    plt.close()

# ---------- 4) confusion matrix ----------
try:
    if preds.size and gts.size:
        n_cls = int(max(preds.max(), gts.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"{dataset} Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        for i, j in itertools.product(range(n_cls), range(n_cls)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dataset}_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
