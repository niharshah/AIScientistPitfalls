import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

# ------------ Plot 1: train / val loss ---------------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    spr = experiment_data.get("SPR", {})
    plt.figure()
    tr = spr.get("losses", {}).get("train", [])
    if tr:
        ep, ls = zip(*tr)
        plt.plot(ep, ls, "--", label="train")
    val = spr.get("losses", {}).get("val", [])
    if val:
        ep, ls = zip(*val)
        plt.plot(ep, ls, "-", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR dataset – Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------ Plot 2: validation metrics -------------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    spr = experiment_data["SPR"]
    val_m = spr.get("metrics", {}).get("val", [])
    if not val_m:
        raise ValueError("No validation metrics to plot.")
    ep = [t[0] for t in val_m]
    cwa = [t[1] for t in val_m]
    swa = [t[2] for t in val_m]
    hm = [t[3] for t in val_m]
    ocg = [t[4] for t in val_m]
    plt.figure()
    plt.plot(ep, cwa, label="CWA")
    plt.plot(ep, swa, label="SWA")
    plt.plot(ep, hm, label="Harmonic Mean")
    plt.plot(ep, ocg, label="OCGA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR dataset – Validation Metrics Over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_validation_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()

# ------------ Plot 3: confusion matrix ---------------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    preds = np.array(experiment_data["SPR"].get("predictions", []))
    gts = np.array(experiment_data["SPR"].get("ground_truth", []))
    if preds.size == 0 or gts.size == 0:
        raise ValueError("Predictions or ground truth missing.")
    classes = np.unique(np.concatenate([gts, preds]))
    n_cls = classes.size
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("SPR dataset – Confusion Matrix (Test Set)")
    plt.xticks(classes)
    plt.yticks(classes)
    fname = os.path.join(working_dir, "SPR_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
