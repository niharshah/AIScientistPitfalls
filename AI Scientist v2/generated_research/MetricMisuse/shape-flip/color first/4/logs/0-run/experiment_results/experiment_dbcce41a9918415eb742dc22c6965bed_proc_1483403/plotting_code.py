import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["no_positional_embedding"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

loss_tr = np.asarray(spr_data["losses"]["train"])
loss_val = np.asarray(spr_data["losses"]["val"])
metrics_val = spr_data["metrics"]["val"]
cwa = np.asarray([m["CWA"] for m in metrics_val])
swa = np.asarray([m["SWA"] for m in metrics_val])
comp = np.asarray([m["CompWA"] for m in metrics_val])
test_metrics = spr_data["metrics"]["test"]
gt = np.asarray(spr_data["ground_truth"])
pred = np.asarray(spr_data["predictions"])

epochs = np.arange(1, len(loss_tr) + 1)

# ---------- plot 1: loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR – Training vs. Validation Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- plot 2: metric curves ----------
try:
    plt.figure()
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, comp, label="CompWA")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("SPR – Validation Metrics over Epochs")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_metric_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric curve plot: {e}")
    plt.close()

# ---------- plot 3: confusion matrix ----------
try:
    from itertools import product

    labels = sorted(set(gt) | set(pred))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for g, p in zip(gt, pred):
        cm[g, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xticks(labels)
    plt.yticks(labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("SPR – Confusion Matrix (Test Set)")
    for i, j in product(range(len(labels)), range(len(labels))):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- plot 4: test metrics bar ----------
try:
    plt.figure()
    names = list(test_metrics.keys())
    vals = [test_metrics[k] for k in names]
    plt.bar(names, vals, color=["tab:blue", "tab:orange", "tab:green"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("SPR – Test Weighted Accuracies")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_test_metrics_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar plot: {e}")
    plt.close()
