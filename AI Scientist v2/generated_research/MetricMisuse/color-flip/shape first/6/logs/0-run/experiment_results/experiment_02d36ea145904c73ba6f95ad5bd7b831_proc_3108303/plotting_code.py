import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp_key = list(experiment_data.keys())[0]  # expected 'contrastive_context_aware'
    data = experiment_data[exp_key]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}


def arr(k1, k2):
    return np.array(data[k1][k2]) if data else np.array([])


# ---------- figure 1 : loss curves ----------
try:
    tr_loss = arr("losses", "train")
    val_loss = arr("losses", "val")
    if tr_loss.size and val_loss.size:
        epochs = np.arange(1, len(tr_loss) + 1)
        plt.figure()
        plt.plot(epochs, tr_loss, "r--", label="Train")
        plt.plot(epochs, val_loss, "r-", label="Validation")
        plt.xlabel("Fine-tuning Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{exp_key} Loss Curves\nLeft: Train (--), Right: Validation (—)")
        plt.legend()
        fname = os.path.join(working_dir, f"{exp_key}_loss_curves.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- figure 2 : metric curves ----------
try:
    swa = arr("metrics", "val_swaa")
    cwa = arr("metrics", "val_cwaa")
    scaa = arr("metrics", "val_scaa")
    if swa.size and cwa.size and scaa.size:
        epochs = np.arange(1, len(swa) + 1)
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, scaa, label="SCAA")
        plt.xlabel("Fine-tuning Epoch")
        plt.ylabel("Metric Value")
        plt.title(f"{exp_key} Validation Metrics Across Epochs\nDataset: SPR")
        plt.legend()
        fname = os.path.join(working_dir, f"{exp_key}_val_metrics.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# ---------- figure 3 : prediction vs ground truth ----------
try:
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))
    if preds.size and gts.size:
        labels = sorted(set(gts) | set(preds))
        pred_counts = [np.sum(preds == l) for l in labels]
        gt_counts = [np.sum(gts == l) for l in labels]
        x = np.arange(len(labels))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_counts, width, label="Predictions")
        plt.xlabel("Class Label")
        plt.ylabel("Count")
        plt.title(f"Label Distribution\n{exp_key}: Ground Truth vs Predictions")
        plt.xticks(x, labels)
        plt.legend()
        fname = os.path.join(working_dir, f"{exp_key}_label_distribution.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Error creating distribution plot: {e}")
    plt.close()

# ---------- print evaluation summary ----------
if cwa.size:
    print(f"Best Validation CWA = {cwa.max():.3f}")
