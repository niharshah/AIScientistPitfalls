import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

spr = exp_data["num_gnn_layers"]["SPR"]
depths = spr["depths"]
train_loss, val_loss = spr["losses"]["train"], spr["losses"]["val"]
train_met, val_met = spr["metrics"]["train"], spr["metrics"]["val"]
test_met = spr["metrics"]["test"]
y_pred, y_true = np.array(spr["predictions"]), np.array(spr["ground_truth"])

# ---------- plot 1: losses ----------
try:
    plt.figure()
    plt.plot(depths, train_loss, "o-", label="Train Loss")
    plt.plot(depths, val_loss, "s-", label="Val Loss")
    plt.title("SPR: Loss vs GNN Depth")
    plt.xlabel("Number of GNN Layers")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_loss_vs_depth.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- plot 2: metrics ----------
try:
    plt.figure()
    plt.plot(depths, train_met, "o-", label="Train CpxWA")
    plt.plot(depths, val_met, "s-", label="Val CpxWA")
    plt.hlines(
        test_met,
        depths[0],
        depths[-1],
        colors="r",
        linestyles="--",
        label=f"Test CpxWA={test_met:.2f}",
    )
    plt.title("SPR: Complexity-Weighted Accuracy vs GNN Depth")
    plt.xlabel("Number of GNN Layers")
    plt.ylabel("CpxWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_metric_vs_depth.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# ---------- plot 3: confusion matrix ----------
try:
    n_cls = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(n_cls))
    plt.yticks(range(n_cls))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SPR: Confusion Matrix (Test Set)")
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fname = os.path.join(working_dir, "SPR_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- print summary ----------
print(f"Test CpxWA: {test_met:.4f}")
print("Confusion matrix:\n", cm)
