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
    experiment_data = {}

model_key = "No-Bidirectional-GRU"
dataset_key = "SPR_BENCH"
ed = experiment_data.get(model_key, {}).get(dataset_key, {})


# Helper to downsample epochs to at most 5 points
def downsample(arr_list, max_pts=5):
    if len(arr_list) <= max_pts:
        return arr_list
    step = max(1, len(arr_list) // max_pts)
    return arr_list[::step]


# 1. Loss curves
try:
    train_loss = downsample(ed["losses"]["train"])
    val_loss = downsample(ed["losses"]["val"])
    tr_epochs, tr_vals = zip(*train_loss)
    va_epochs, va_vals = zip(*val_loss)
    plt.figure()
    plt.plot(tr_epochs, tr_vals, label="Train Loss")
    plt.plot(va_epochs, va_vals, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2. Validation metric curves
try:
    metrics = downsample(ed["metrics"]["val"])
    epochs, cwa, swa, hcs, snwa = zip(*metrics)
    plt.figure()
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, hcs, label="HCSA")
    plt.plot(epochs, snwa, label="SNWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH Validation Metrics")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_validation_metrics.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating metric curves: {e}")
    plt.close()


# Confusion-matrix helper
def plot_conf_mat(gts, preds, split_name):
    try:
        num_cls = int(max(max(gts), max(preds))) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for g, p in zip(gts, preds):
            cm[g, p] += 1
        plt.figure(figsize=(5, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"SPR_BENCH Confusion Matrix - {split_name}")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        # annotate counts
        for i in range(num_cls):
            for j in range(num_cls):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        fname = os.path.join(working_dir, f"SPR_BENCH_conf_mat_{split_name}.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating confusion matrix ({split_name}): {e}")
        plt.close()


# 3. Dev confusion matrix
plot_conf_mat(
    ed.get("ground_truth", {}).get("dev", []),
    ed.get("predictions", {}).get("dev", []),
    "Dev",
)

# 4. Test confusion matrix
plot_conf_mat(
    ed.get("ground_truth", {}).get("test", []),
    ed.get("predictions", {}).get("test", []),
    "Test",
)
