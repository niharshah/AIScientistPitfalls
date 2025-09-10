import matplotlib.pyplot as plt
import numpy as np
import os

# ensure working directory exists
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# quick guard
if not experiment_data:
    print("No experiment data found; nothing to plot.")
    exit()

run = experiment_data["no_learned_pos_emb"]["SPR_BENCH"]
train_loss = run["losses"]["train"]
val_loss = run["losses"]["val"]
metrics = run["metrics"]["val"]  # list of dicts
macro_f1 = [m["macro_f1"] for m in metrics]
cwa = [m["cwa"] for m in metrics]
preds = np.array(run["predictions"])
labels = np.array(run["ground_truth"])
weights = np.array(run["weights"])
epochs = np.arange(1, len(train_loss) + 1)


# helper for epoch thinning (max 5 markers)
def idx_subset(x, n=5):
    if len(x) <= n:
        return np.arange(len(x))
    step = max(1, len(x) // n)
    return np.arange(0, len(x), step)[:n]


# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.scatter(
        epochs[idx_subset(epochs)], np.array(train_loss)[idx_subset(epochs)], c="blue"
    )
    plt.scatter(
        epochs[idx_subset(epochs)], np.array(val_loss)[idx_subset(epochs)], c="orange"
    )
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2) Metric curves
try:
    plt.figure()
    plt.plot(epochs, macro_f1, label="Macro-F1")
    plt.plot(epochs, cwa, label="CWA")
    plt.scatter(
        epochs[idx_subset(epochs)], np.array(macro_f1)[idx_subset(epochs)], c="green"
    )
    plt.scatter(epochs[idx_subset(epochs)], np.array(cwa)[idx_subset(epochs)], c="red")
    plt.title("SPR_BENCH: Validation Metrics over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_metric_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric curve plot: {e}")
    plt.close()

# 3) Histogram of weights
try:
    plt.figure()
    plt.hist(weights, bins=20, color="purple", alpha=0.7)
    plt.title("SPR_BENCH: Distribution of Instance Weights")
    plt.xlabel("Weight")
    plt.ylabel("Count")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_weight_histogram.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating weight histogram: {e}")
    plt.close()

# 4) Weight vs Correctness scatter
try:
    correctness = preds == labels
    plt.figure()
    plt.scatter(
        weights[correctness],
        np.zeros_like(weights[correctness]),
        c="green",
        label="Correct",
        alpha=0.5,
    )
    plt.scatter(
        weights[~correctness],
        np.zeros_like(weights[~correctness]),
        c="red",
        label="Incorrect",
        alpha=0.5,
    )
    plt.yticks([])
    plt.title("SPR_BENCH: Weight vs Prediction Correctness")
    plt.xlabel("Instance Weight")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_weight_correct_scatter.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating weight-correctness scatter: {e}")
    plt.close()

# 5) Confusion matrix
try:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds, labels=sorted(set(labels)))
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title("SPR_BENCH: Confusion Matrix (Validation)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------- print final metrics ---------------
if macro_f1:
    print(f"Final Macro-F1: {macro_f1[-1]:.3f}")
    print(f"Final CWA: {cwa[-1]:.3f}")
