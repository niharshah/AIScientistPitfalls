import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------- setup -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ------------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    spr_data = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = {}

# ---------------- metrics --------------------
loss_train = np.array(spr_data.get("losses", {}).get("train", []))
loss_val = np.array(spr_data.get("losses", {}).get("val", []))
cwa_val = np.array(spr_data.get("metrics", {}).get("val", []))
preds = np.array(spr_data.get("predictions", []))
gtruth = np.array(spr_data.get("ground_truth", []))

acc = float("nan")
if preds.size and gtruth.size:
    acc = (preds == gtruth).mean()
best_cwa = cwa_val.max() if cwa_val.size else float("nan")
print(f"Overall accuracy (validation best epoch): {acc:.4f}")
print(f"Best validation CWA-2D: {best_cwa:.4f}")

# --------------- 1. loss curves --------------
try:
    if loss_train.size and loss_val.size:
        plt.figure()
        epochs = np.arange(1, len(loss_train) + 1)
        plt.plot(epochs, loss_train, label="Train")
        plt.plot(epochs, loss_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------------- 2. CWA over epochs ----------
try:
    if cwa_val.size:
        plt.figure()
        epochs = np.arange(1, len(cwa_val) + 1)
        plt.plot(epochs, cwa_val, marker="o", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("CWA-2D")
        plt.title("SPR_BENCH – Validation CWA-2D per Epoch")
        fname = os.path.join(working_dir, "SPR_BENCH_CWA_per_epoch.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# ------------- 3. confusion matrix ----------
try:
    if preds.size and gtruth.size:
        num_cls = int(max(preds.max(), gtruth.max())) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gtruth, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted label")
        plt.ylabel("Ground Truth label")
        plt.title("SPR_BENCH – Confusion Matrix\n(rows = GT, cols = Pred)")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
