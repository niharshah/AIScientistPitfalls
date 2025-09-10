import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------- #
# Load experiment data
# ---------------------------------------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to fetch nested dict safely
def get_nested(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


exp_path = ["Frozen-Cluster-Embeddings", "SPR_BENCH"]
loss_train = get_nested(experiment_data, exp_path + ["losses", "train"], [])
loss_val = get_nested(experiment_data, exp_path + ["losses", "val"], [])
metrics_val = get_nested(experiment_data, exp_path + ["metrics", "val"], [])
preds_test = get_nested(experiment_data, exp_path + ["predictions", "test"], [])
gts_test = get_nested(experiment_data, exp_path + ["ground_truth", "test"], [])

# ---------------------------------------------------- #
# 1. Loss curves
# ---------------------------------------------------- #
try:
    if loss_train and loss_val:
        ep_t, l_t = zip(*loss_train)
        ep_v, l_v = zip(*loss_val)
        plt.figure()
        plt.plot(ep_t, l_t, label="Train")
        plt.plot(ep_v, l_v, label="Validation")
        plt.title("SPR_BENCH Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    else:
        print("Loss data missing, skipping loss curve.")
except Exception as e:
    print(f"Error creating loss curve: {e}")
finally:
    plt.close()

# ---------------------------------------------------- #
# 2. Validation metric curves (HCSA & SNWA)
# ---------------------------------------------------- #
try:
    if metrics_val:
        ep, cwa, swa, hcs, snwa = zip(*metrics_val)
        plt.figure()
        plt.plot(ep, hcs, label="HCSA")
        plt.plot(ep, snwa, label="SNWA")
        plt.title("SPR_BENCH Validation Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_metrics_curve.png"))
    else:
        print("Metric data missing, skipping metric curve.")
except Exception as e:
    print(f"Error creating metric curve: {e}")
finally:
    plt.close()

# ---------------------------------------------------- #
# 3. Confusion matrix on test set
# ---------------------------------------------------- #
try:
    if preds_test and gts_test:
        preds = np.array(preds_test)
        gts = np.array(gts_test)
        n_cls = max(preds.max(), gts.max()) + 1
        cm, _, _ = np.histogram2d(gts, preds, bins=[np.arange(n_cls + 1)] * 2)
        plt.figure()
        im = plt.imshow(cm, interpolation="nearest", cmap="viridis")
        plt.title("SPR_BENCH Confusion Matrix (Test Set)")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    else:
        print("Prediction data missing, skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
finally:
    plt.close()
