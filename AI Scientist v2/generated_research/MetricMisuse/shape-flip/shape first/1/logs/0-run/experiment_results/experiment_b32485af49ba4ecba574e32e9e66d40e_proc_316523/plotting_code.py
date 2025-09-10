import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter, defaultdict

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

# ---------- helpers ----------
epochs = list(range(1, 1 + len(exp.get("metrics", {}).get("train_acc", []))))

# ------------------ FIG 1: accuracy curves ---------------------------
try:
    plt.figure()
    tr_acc = exp["metrics"]["train_acc"]
    val_acc = exp["metrics"]["val_acc"]
    plt.plot(epochs, tr_acc, marker="o", label="Train")
    plt.plot(epochs, val_acc, marker="x", linestyle="--", label="Validation")
    plt.title("SPR_BENCH: Train & Val Accuracy vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# ------------------ FIG 2: loss curves --------------------------------
try:
    plt.figure()
    tr_loss = exp["losses"]["train"]
    val_loss = exp["losses"]["val"]
    plt.plot(epochs, tr_loss, marker="o", label="Train")
    plt.plot(epochs, val_loss, marker="x", linestyle="--", label="Validation")
    plt.title("SPR_BENCH: Train & Val Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------ FIG 3: test metrics bar ---------------------------
try:
    plt.figure()
    acc = exp["metrics"]["val_acc"][-1] if exp["metrics"]["val_acc"] else np.nan
    swa = exp["metrics"]["SWA"][0] if exp["metrics"]["SWA"] else np.nan
    ura = exp["metrics"]["URA"][0] if exp["metrics"]["URA"] else np.nan
    metrics = [acc, swa, ura]
    names = ["Accuracy", "SWA", "URA"]
    plt.bar(names, metrics, color=["skyblue", "lightgreen", "salmon"])
    plt.title("SPR_BENCH: Final Test Metrics")
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics_bar.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar plot: {e}")
    plt.close()

# ------------------ FIG 4: confusion matrix (top 10 labels) ----------
try:
    preds = exp.get("predictions", [])
    gts = exp.get("ground_truth", [])
    if preds and gts:
        # determine top 10 labels by occurrence in ground truth
        top_labels = [l for l, _ in Counter(gts).most_common(10)]
        idx_map = {l: i for i, l in enumerate(top_labels)}
        cm = np.zeros((len(top_labels), len(top_labels)), dtype=int)
        for gt, pr in zip(gts, preds):
            if gt in idx_map and pr in idx_map:
                cm[idx_map[gt], idx_map[pr]] += 1
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(top_labels)), top_labels, rotation=90)
        plt.yticks(range(len(top_labels)), top_labels)
        plt.title(
            "SPR_BENCH: Confusion Matrix (Top-10 Labels)\nLeft: Ground Truth, Top: Predicted"
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("Ground Truth Label")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix_top10.png")
        plt.tight_layout()
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
