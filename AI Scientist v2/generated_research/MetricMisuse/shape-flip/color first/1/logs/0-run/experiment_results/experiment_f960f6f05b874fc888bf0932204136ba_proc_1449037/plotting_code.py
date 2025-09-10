import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# assume the single run present
model_key = next(iter(experiment_data.keys()), None)
ds_key = "SPR"
if model_key is None or ds_key not in experiment_data.get(model_key, {}):
    print("No experiment data found, aborting plots.")
    exit()

run = experiment_data[model_key][ds_key]
epochs = run["epochs"]

# ---------- plot 1: loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, run["losses"]["train"], label="Train")
    plt.plot(epochs, run["losses"]["val"], label="Validation")
    plt.title(f"SPR Loss Curves ({model_key})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- plot 2: validation metrics ----------
try:
    cwa = [d["CWA"] for d in run["metrics"]["val"]]
    swa = [d["SWA"] for d in run["metrics"]["val"]]
    hpa = [d["HPA"] for d in run["metrics"]["val"]]
    plt.figure()
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, hpa, label="HPA")
    plt.title(f"SPR Validation Metrics ({model_key})")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend()
    fname = os.path.join(working_dir, "SPR_validation_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# ---------- plot 3: confusion matrix ----------
try:
    preds = np.array(run["predictions"])
    gts = np.array(run["ground_truth"])
    n_cls = len(set(gts) | set(preds))
    cm = np.zeros((n_cls, n_cls), int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.title("SPR Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(label="Count")
    fname = os.path.join(working_dir, "SPR_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- final metrics ----------
try:
    # recompute in case they aren't in the dicts
    def count_color_variety(seq):
        return len(set(tok[1] for tok in seq.split() if len(tok) > 1))

    def count_shape_variety(seq):
        return len(set(tok[0] for tok in seq.split() if tok))

    seqs = [
        g for g in experiment_data[model_key][ds_key].get("sequences", [])
    ]  # may be missing
    if not seqs:
        print("Sequences not stored, skipping metric recomputation.")
    else:
        w_c = [count_color_variety(s) for s in seqs]
        w_s = [count_shape_variety(s) for s in seqs]
        cwa = sum(w if t == p else 0 for w, t, p in zip(w_c, gts, preds)) / max(
            sum(w_c), 1
        )
        swa = sum(w if t == p else 0 for w, t, p in zip(w_s, gts, preds)) / max(
            sum(w_s), 1
        )
        hpa = 2 * cwa * swa / (cwa + swa + 1e-8)
        print(f"Test metrics  |  CWA={cwa:.3f}  SWA={swa:.3f}  HPA={hpa:.3f}")
except Exception as e:
    print(f"Error computing final metrics: {e}")
