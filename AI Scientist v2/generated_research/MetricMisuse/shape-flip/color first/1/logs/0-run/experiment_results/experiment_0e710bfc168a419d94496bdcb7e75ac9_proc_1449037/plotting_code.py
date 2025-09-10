import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("shape_only", {}).get("SPR", {})
losses = spr.get("losses", {})
metrics = spr.get("metrics", {})
epochs = spr.get("epochs", [])
preds = np.array(spr.get("predictions", []))
gts = np.array(spr.get("ground_truth", []))


# ---------------- helpers -------------------
def safe_len(x):
    return len(x) if isinstance(x, (list, tuple)) else 0


# --------- 1) loss curve --------------------
try:
    if safe_len(losses.get("train", [])) and safe_len(losses.get("val", [])):
        plt.figure()
        plt.plot(epochs, losses["train"], label="Train Loss")
        plt.plot(epochs, losses["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Dataset – Shape-only Model\nTraining vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curve_shape_only.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# --------- 2) validation metrics ------------
try:
    if safe_len(metrics.get("val", [])):
        cwa = [m["CWA"] for m in metrics["val"]]
        swa = [m["SWA"] for m in metrics["val"]]
        hpa = [m["HPA"] for m in metrics["val"]]
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, hpa, label="HPA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR Dataset – Shape-only Model\nValidation Metrics Over Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_val_metrics_shape_only.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating metric curve: {e}")
    plt.close()

# --------- 3) prediction vs ground-truth ----
try:
    if preds.size and gts.size:
        classes = sorted(set(np.concatenate([preds, gts])))
        pred_counts = [int((preds == c).sum()) for c in classes]
        gt_counts = [int((gts == c).sum()) for c in classes]

        x = np.arange(len(classes))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_counts, width, label="Predictions")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("SPR Dataset – Shape-only Model\nTest Predictions vs Ground Truth")
        plt.xticks(x, classes)
        plt.legend()
        fname = os.path.join(working_dir, "SPR_test_pred_vs_gt_shape_only.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating prediction bar chart: {e}")
    plt.close()

# ------------- print evaluation metric -----
if preds.size and gts.size:
    acc = (preds == gts).mean()
    print(f"Test Accuracy: {acc:.3f}")
