import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------- load experiment data ----------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# helper shortcuts
exp = experiment_data.get("edge_type_collapsed", {}).get("SPR", {})
train_loss = exp.get("losses", {}).get("train", [])
val_loss = exp.get("losses", {}).get("val", [])
metrics = exp.get("metrics", {}).get("val", [])
epochs = exp.get("epochs", [])
preds = np.array(exp.get("predictions", []))
gts = np.array(exp.get("ground_truth", []))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def cwa(seqs, y, p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, pt in zip(w, y, p) if yt == pt) / max(sum(w), 1)


def swa(seqs, y, p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, pt in zip(w, y, p) if yt == pt) / max(sum(w), 1)


def hpa(c, s):
    return 2 * c * s / (c + s + 1e-8)


# ----------------------------- plots ------------------------------
# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Dataset – Training vs. Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_loss_curves_edge_type_collapsed.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Metric curves (CWA/SWA/HPA)
try:
    if metrics:
        c = [m["CWA"] for m in metrics]
        s = [m["SWA"] for m in metrics]
        h = [m["HPA"] for m in metrics]
        plt.figure()
        plt.plot(epochs, c, label="CWA")
        plt.plot(epochs, s, label="SWA")
        plt.plot(epochs, h, label="HPA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR Dataset – Validation Metrics Over Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_val_metrics_edge_type_collapsed.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# 3) Confusion matrix heat-map
try:
    if preds.size and gts.size:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title("SPR Dataset – Test Confusion Matrix")
        fname = os.path.join(
            working_dir, "SPR_confusion_matrix_edge_type_collapsed.png"
        )
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------------------- print final test metrics ------------------
try:
    seqs_placeholder = [""] * len(preds)  # sequences not saved; metrics need them
    C = cwa(seqs_placeholder, gts, preds)
    S = swa(seqs_placeholder, gts, preds)
    H = hpa(C, S)
    print(f"Final TEST metrics | CWA={C:.3f}  SWA={S:.3f}  HPA={H:.3f}")
except Exception as e:
    print(f"Error computing final metrics: {e}")
