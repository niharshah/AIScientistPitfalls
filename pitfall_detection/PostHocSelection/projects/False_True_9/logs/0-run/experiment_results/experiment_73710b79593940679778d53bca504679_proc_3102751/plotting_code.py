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
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("SPR_BENCH", {})


# ---------- helper to slice lists ----------
def unzip(pairs):
    return zip(*pairs) if pairs else ([], [])


# ---------- 1) loss curves ---------------
try:
    tr_epochs, tr_vals = unzip(spr.get("losses", {}).get("train", []))
    val_epochs, val_vals = unzip(spr.get("losses", {}).get("val", []))
    if tr_epochs and val_epochs:
        plt.figure()
        plt.plot(tr_epochs, tr_vals, label="Train")
        plt.plot(val_epochs, val_vals, label="Validation", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- 2) metric curves --------------
try:
    metric_arr = spr.get("metrics", {}).get("val", [])
    ep, swa, cwa, dawa = unzip(metric_arr)
    if ep:
        plt.figure()
        plt.plot(ep, swa, label="SWA")
        plt.plot(ep, cwa, label="CWA")
        plt.plot(ep, dawa, label="DAWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Validation Metrics Over Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_metric_curves.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# ---------- 3) confusion matrix ----------
try:
    preds = spr.get("predictions", [])
    gts = spr.get("ground_truth", [])
    if preds and gts:
        num_cls = max(max(preds), max(gts)) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH: Confusion Matrix (Final Epoch)")
        plt.xticks(range(num_cls))
        plt.yticks(range(num_cls))
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- print summary -----------------
if spr.get("metrics", {}).get("val"):
    last_dawa = spr["metrics"]["val"][-1][-1]
    print(f"Final-epoch DAWA: {last_dawa:.4f}")
