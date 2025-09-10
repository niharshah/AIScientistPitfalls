import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment dictionary
# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
if dataset not in experiment_data:
    print(f"Dataset {dataset} not found in experiment_data.")
    exit()

data = experiment_data[dataset]
epochs = data.get("epochs", [])
train_losses = data.get("losses", {}).get("train", [])
val_losses = data.get("losses", {}).get("val", [])
val_metrics = data.get("metrics", {}).get("val", [])
preds = np.array(data.get("predictions", []))
gts = np.array(data.get("ground_truth", []))
test_metrics = data.get("metrics", {}).get("test", {})


# ------------------------------------------------------------------
# Helper to extract metric arrays
# ------------------------------------------------------------------
def metric_list(mname):
    return [m.get(mname, np.nan) for m in val_metrics]


# ------------------------------------------------------------------
# 1. Loss curves
# ------------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{dataset} – Training vs. Validation Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset}_loss_curve.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2. Validation metric curves
# ------------------------------------------------------------------
try:
    plt.figure()
    for m in ["CWA", "SWA", "CpxWA"]:
        plt.plot(epochs, metric_list(m), label=m)
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title(f"{dataset} – Validation Weighted Accuracies")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset}_val_metrics.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating metric curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3. Per-class accuracy bar chart
# ------------------------------------------------------------------
try:
    if preds.size and gts.size:
        num_classes = int(max(gts.max(), preds.max()) + 1)
        acc_per_class = []
        for c in range(num_classes):
            idx = gts == c
            acc = (preds[idx] == gts[idx]).mean() if idx.any() else np.nan
            acc_per_class.append(acc)
        plt.figure(figsize=(max(6, num_classes * 0.3), 4))
        plt.bar(range(num_classes), acc_per_class)
        plt.xlabel("Class Index")
        plt.ylabel("Accuracy")
        plt.title(f"{dataset} – Per-Class Accuracy (Test Set)")
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dataset}_per_class_accuracy.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating per-class accuracy chart: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print final test metrics
# ------------------------------------------------------------------
if test_metrics:
    print(
        f"FINAL TEST METRICS – CWA: {test_metrics.get('CWA'):.4f}, "
        f"SWA: {test_metrics.get('SWA'):.4f}, "
        f"CpxWA: {test_metrics.get('CpxWA'):.4f}"
    )
