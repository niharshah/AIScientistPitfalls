import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# ------------------------------------------------------------------
# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helpers
def unpack(k, split):
    """Return epochs and values arrays for key k ('losses' or 'metrics') / split."""
    data = experiment_data["pooling_method"][pm][k][split]
    if not data:
        return np.array([]), np.array([])
    ep, val = zip(*data)
    return np.array(ep), np.array(val)


# ------------------------------------------------------------------
# 1) Loss curves (train vs val) for all pooling methods
try:
    plt.figure()
    for pm in experiment_data.get("pooling_method", {}):
        if pm in ("best_pooling", "predictions", "ground_truth"):
            continue
        ep_tr, loss_tr = unpack("losses", "train")
        ep_vl, loss_vl = unpack("losses", "val")
        if ep_tr.size and ep_vl.size:
            plt.plot(ep_tr, loss_tr, label=f"{pm}-train")
            plt.plot(ep_vl, loss_vl, linestyle="--", label=f"{pm}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR dataset – Loss Curves (Train vs Validation)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_loss_curves.png")
    plt.savefig(fname)
    print("Saved ->", fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Harmonic-weighted accuracy curves
try:
    plt.figure()
    for pm in experiment_data.get("pooling_method", {}):
        if pm in ("best_pooling", "predictions", "ground_truth"):
            continue
        ep_tr, hwa_tr = unpack("metrics", "train")
        ep_vl, hwa_vl = unpack("metrics", "val")
        if ep_tr.size and ep_vl.size:
            plt.plot(ep_tr, hwa_tr, label=f"{pm}-train")
            plt.plot(ep_vl, hwa_vl, linestyle="--", label=f"{pm}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic-Weighted Accuracy")
    plt.title("SPR dataset – HWA Curves (Train vs Validation)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_hwa_curves.png")
    plt.savefig(fname)
    print("Saved ->", fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Confusion matrix for best pooling method
try:
    best_pool = experiment_data["pooling_method"].get("best_pooling", None)
    preds = np.array(experiment_data["pooling_method"].get("predictions", []))
    trues = np.array(experiment_data["pooling_method"].get("ground_truth", []))
    if best_pool is not None and preds.size and trues.size:
        conf = np.zeros((2, 2), int)
        for t, p in zip(trues, preds):
            conf[t, p] += 1
        plt.figure()
        im = plt.imshow(conf, cmap="Blues", vmin=0)
        plt.colorbar(im)
        plt.xticks([0, 1], ["0", "1"])
        plt.yticks([0, 1], ["0", "1"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"SPR dataset – Confusion Matrix (best pooling: {best_pool})")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, conf[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, f"SPR_confusion_best_{best_pool}.png")
        plt.savefig(fname)
        print("Saved ->", fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
