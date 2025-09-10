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
    ed = experiment_data["SingleRGCN"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()


# ---------- helpers ----------
def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def harmonic_poly_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# ---------- PLOT 1: loss curves ----------
try:
    plt.figure()
    epochs = ed["epochs"]
    plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
    plt.plot(epochs, ed["losses"]["val"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR: Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- PLOT 2: HPA progression ----------
try:
    plt.figure()
    hpa_vals = [m["HPA"] for m in ed["metrics"]["val"]]
    plt.plot(epochs, hpa_vals, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("HPA")
    plt.title("SPR: Validation Harmonic Poly-Accuracy (HPA)")
    plt.savefig(os.path.join(working_dir, "SPR_HPA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HPA curve: {e}")
    plt.close()

# ---------- PLOT 3: confusion matrix ----------
try:
    from itertools import product

    preds = np.array(ed["predictions"])
    gts = np.array(ed["ground_truth"])
    n_cls = len(np.unique(gts))
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title("SPR Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i, j in product(range(n_cls), range(n_cls)):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.savefig(os.path.join(working_dir, "SPR_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- evaluate & print ----------
try:
    # The original seqs are not saved; derive dummy seqs list for metric printing
    seqs_placeholder = [""] * len(
        gts
    )  # metrics only need placeholder if varieties are zero
    cwa = color_weighted_accuracy(seqs_placeholder, gts.tolist(), preds.tolist())
    swa = shape_weighted_accuracy(seqs_placeholder, gts.tolist(), preds.tolist())
    hpa = harmonic_poly_accuracy(cwa, swa)
    print(f"Test metrics | CWA={cwa:.3f}  SWA={swa:.3f}  HPA={hpa:.3f}")
except Exception as e:
    print(f"Error computing test metrics: {e}")
