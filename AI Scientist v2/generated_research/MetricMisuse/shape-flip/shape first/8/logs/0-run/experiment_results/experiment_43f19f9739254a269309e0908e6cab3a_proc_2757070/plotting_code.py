import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to safely fetch nested dict keys
def deep_get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# With only one dataset/setting we can fetch it directly
ds_name = next(iter(experiment_data.keys()), None)
setting_name = next(iter(experiment_data.get(ds_name, {}).keys()), None)

metrics = deep_get(experiment_data, ds_name, setting_name, "metrics", default={})
losses = deep_get(experiment_data, ds_name, setting_name, "losses", default={})
preds = deep_get(experiment_data, ds_name, setting_name, "predictions", default=[])
gts = deep_get(experiment_data, ds_name, setting_name, "ground_truth", default=[])

epochs = np.arange(1, len(losses.get("train", [])) + 1)

# ------------------------------------------------------------------
# 1. Loss curves
try:
    plt.figure()
    plt.plot(epochs, losses.get("train", []), label="Train")
    plt.plot(epochs, losses.get("val", []), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{ds_name} – Loss Curves")
    plt.legend()
    fname = f"{ds_name}_loss_curves.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2. Shape-weighted accuracy curves
try:
    plt.figure()
    plt.plot(epochs, metrics.get("train_swa", []), label="Train")
    plt.plot(epochs, metrics.get("val_swa", []), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title(f"{ds_name} – Shape-Weighted Accuracy")
    plt.legend()
    fname = f"{ds_name}_swa_curves.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3. Class-count bar chart (Ground Truth vs Predictions)
try:
    gt_arr = np.array(gts)
    pr_arr = np.array(preds)
    classes = [0, 1]
    counts_gt = [np.sum(gt_arr == c) for c in classes]
    counts_pr = [np.sum(pr_arr == c) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, counts_gt, width, label="Ground Truth")
    plt.bar(x + width / 2, counts_pr, width, label="Predictions")
    plt.xticks(x, [str(c) for c in classes])
    plt.ylabel("Count")
    plt.title(
        f"{ds_name} – Test Set Class Distribution\nLeft: Ground Truth, Right: Model Predictions"
    )
    plt.legend()
    fname = f"{ds_name}_class_distribution.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating class distribution plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 4. Confusion matrix heat-map
try:
    cm = np.zeros((2, 2), dtype=int)
    for gt, pr in zip(gt_arr, pr_arr):
        cm[gt, pr] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.title(f"{ds_name} – Confusion Matrix (Test Set)")
    fname = f"{ds_name}_confusion_matrix.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
