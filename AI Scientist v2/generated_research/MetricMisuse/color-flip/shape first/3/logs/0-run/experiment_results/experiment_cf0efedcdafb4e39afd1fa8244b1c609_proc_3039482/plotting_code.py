import matplotlib.pyplot as plt
import numpy as np
import os

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

variant = "mask_only"
dataset = "SPR_BENCH"
rec = experiment_data.get(variant, {}).get(dataset, {})


# Helper to safely grab arrays
def get(path, default=None):
    d = rec
    for k in path.split("."):
        d = d.get(k, {})
    return d if isinstance(d, (list, tuple)) else default


# 1) Contrastive pretraining loss
try:
    pre_loss = get("losses.pretrain", [])
    if pre_loss:
        plt.figure()
        plt.plot(range(1, len(pre_loss) + 1), pre_loss, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset} – Contrastive Pre-training Loss", fontsize=12)
        plt.suptitle("Mask-only augmentation (Left: Epochs, Right: Loss)", fontsize=9)
        fname = os.path.join(working_dir, f"{dataset}_pretrain_loss.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating pretrain plot: {e}")
    plt.close()

# 2) Fine-tuning train/val loss
try:
    tr_loss = get("losses.train", [])
    val_loss = get("losses.val", [])
    if tr_loss and val_loss:
        plt.figure()
        epochs = range(1, len(tr_loss) + 1)
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"{dataset} – Fine-tuning Loss Curves", fontsize=12)
        plt.suptitle("Left: Train Loss, Right: Validation Loss", fontsize=9)
        fname = os.path.join(working_dir, f"{dataset}_train_val_loss.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating train/val loss plot: {e}")
    plt.close()

# 3) Validation accuracy & ACA
try:
    val_acc = get("metrics.val_acc", [])
    val_aca = get("metrics.val_aca", [])
    if val_acc and val_aca:
        plt.figure()
        epochs = range(1, len(val_acc) + 1)
        plt.plot(epochs, val_acc, label="Val Acc")
        plt.plot(epochs, val_aca, label="Val ACA")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.ylim(0, 1)
        plt.legend()
        plt.title(f"{dataset} – Validation Metrics", fontsize=12)
        plt.suptitle("Left: Accuracy, Right: ACA", fontsize=9)
        fname = os.path.join(working_dir, f"{dataset}_val_metrics.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()

# 4) Test metrics bar chart
try:
    test = rec.get("test", {})
    labels = ["acc", "swa", "cwa", "aca"]
    vals = [test.get(k, np.nan) for k in labels]
    if any(np.isfinite(vals)):
        plt.figure()
        plt.bar(labels, vals)
        plt.ylim(0, 1)
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
        plt.title(f"{dataset} – Test Metrics Summary", fontsize=12)
        plt.suptitle("Acc, Shape-weighted Acc, Color-weighted Acc, ACA", fontsize=9)
        fname = os.path.join(working_dir, f"{dataset}_test_metrics.png")
        plt.savefig(fname)
        plt.close()
        print("Test metrics:", {k: round(test.get(k, np.nan), 4) for k in labels})
except Exception as e:
    print(f"Error creating test metrics plot: {e}")
    plt.close()
