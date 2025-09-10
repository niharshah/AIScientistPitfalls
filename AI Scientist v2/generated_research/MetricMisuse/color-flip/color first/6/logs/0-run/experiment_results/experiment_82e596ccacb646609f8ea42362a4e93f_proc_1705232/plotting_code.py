import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- basic setup ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "shared_embedding"
ds = experiment_data.get(ds_name, {})


# ---------------- helper metrics -------------
def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def cwa(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def swa(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def pcwa(seqs, y_t, y_p):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------------- plotting -------------------
# Plot 1: training & validation loss
try:
    plt.figure()
    tr = ds.get("losses", {}).get("train", [])
    vl = ds.get("losses", {}).get("val", [])
    if tr and vl:
        epochs_tr, loss_tr = zip(*tr)
        epochs_vl, loss_vl = zip(*vl)
        plt.plot(epochs_tr, loss_tr, label="Train Loss")
        plt.plot(epochs_vl, loss_vl, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Shared Embedding SPR Dataset\nTraining vs. Validation Loss")
        plt.legend()
    plt.savefig(os.path.join(working_dir, "shared_embedding_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# Plot 2: validation metrics over epochs
try:
    plt.figure()
    metrics = ds.get("metrics", {}).get("val", [])
    if metrics:
        ep, cwa_v, swa_v, pcwa_v = [], [], [], []
        for e, m in metrics:
            ep.append(e)
            cwa_v.append(m["CWA"])
            swa_v.append(m["SWA"])
            pcwa_v.append(m["PCWA"])
        plt.plot(ep, cwa_v, label="CWA")
        plt.plot(ep, swa_v, label="SWA")
        plt.plot(ep, pcwa_v, label="PCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(
            "Shared Embedding SPR Dataset\nWeighted Accuracy Metrics (Validation)"
        )
        plt.legend()
    plt.savefig(os.path.join(working_dir, "shared_embedding_metrics_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics curve: {e}")
    plt.close()

# ---------------- evaluation on test split ----
try:
    preds = ds.get("predictions", [])
    gtruth = ds.get("ground_truth", [])
    # For these metrics we also need sequences; if unavailable, skip computation
    seqs = ds.get("sequences_test", [])  # not saved by training script
    if preds and gtruth and seqs:
        print("Test CWA:", cwa(seqs, gtruth, preds))
        print("Test SWA:", swa(seqs, gtruth, preds))
        print("Test PCWA:", pcwa(seqs, gtruth, preds))
    elif preds and gtruth:
        acc = sum(p == t for p, t in zip(preds, gtruth)) / len(preds)
        print("Test Accuracy:", acc)
except Exception as e:
    print(f"Error computing evaluation metrics: {e}")
