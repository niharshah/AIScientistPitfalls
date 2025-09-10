import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load -----------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to safely fetch nested keys
def get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


run_key = "UniGRU_no_bidi"
loss_tr = get(experiment_data, run_key, "SPR", "losses", "train", default=[])
loss_val = get(experiment_data, run_key, "SPR", "losses", "val", default=[])
met_val = get(experiment_data, run_key, "SPR", "metrics", "val", default=[])
preds = get(experiment_data, run_key, "SPR", "predictions", default=[])
gts = get(experiment_data, run_key, "SPR", "ground_truth", default=[])

# ---------- 1. loss curves ------------------------------------------
try:
    if loss_tr and loss_val:
        ep_tr, v_tr = zip(*loss_tr)
        ep_val, v_val = zip(*loss_val)
        plt.figure()
        plt.plot(ep_tr, v_tr, label="Train")
        plt.plot(ep_val, v_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Dataset – Training / Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curve_UniGRU_no_bidi.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- 2. validation metrics curves ----------------------------
try:
    if met_val:
        ep, cwa, swa, hm, ocga = zip(*met_val)
        plt.figure()
        plt.plot(ep, cwa, label="CWA")
        plt.plot(ep, swa, label="SWA")
        plt.plot(ep, hm, label="Harmonic Mean")
        plt.plot(ep, ocga, label="OCGA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR Dataset – Validation Metrics over Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_metric_curves_UniGRU_no_bidi.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating metric curves: {e}")
    plt.close()

# ---------- 3. confusion matrix -------------------------------------
try:
    if preds and gts:
        preds = np.asarray(preds)
        gts = np.asarray(gts)
        n_cls = max(preds.max(), gts.max()) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR Dataset – Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, "SPR_confusion_matrix_UniGRU_no_bidi.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- 4. prediction vs ground truth class counts --------------
try:
    if preds and gts:
        uniq = np.arange(max(max(preds), max(gts)) + 1)
        pred_cnt = [(preds == u).sum() for u in uniq]
        gt_cnt = [(gts == u).sum() for u in uniq]
        x = np.arange(len(uniq))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, gt_cnt, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_cnt, width, label="Predicted")
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.title("SPR Dataset – Class Distribution (Test Set)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_class_distribution_UniGRU_no_bidi.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating class distribution plot: {e}")
    plt.close()
