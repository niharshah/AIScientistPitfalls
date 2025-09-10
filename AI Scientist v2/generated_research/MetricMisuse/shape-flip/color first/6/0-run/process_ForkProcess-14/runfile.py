import matplotlib.pyplot as plt
import numpy as np
import os
import random

# ---------- set up working dir ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ---------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("spr_bench", {})
epochs = ed.get("epochs", [])
train_loss = ed.get("losses", {}).get("train", [])
val_loss = ed.get("losses", {}).get("val", [])
val_cplx = ed.get("metrics", {}).get("val", [])
test_cplx = ed.get("metrics", {}).get("test", None)
gt = ed.get("ground_truth", [])
pred = ed.get("predictions", [])

# ---------- Loss curve --------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.title("SPR_BENCH – Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
    plt.savefig(fname)
except Exception as e:
    print(f"Error creating loss curve: {e}")
finally:
    plt.close()

# ---------- Validation CplxWA curve ------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, val_cplx)
    plt.title("SPR_BENCH – Validation Complexity-Weighted Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("CplxWA")
    fname = os.path.join(working_dir, "spr_bench_val_cplxwa_curve.png")
    plt.savefig(fname)
except Exception as e:
    print(f"Error creating CplxWA curve: {e}")
finally:
    plt.close()

# ---------- Prediction vs Ground-Truth scatter -------------------------------
try:
    if gt and pred:
        idx = random.sample(range(len(gt)), min(100, len(gt)))
        plt.figure()
        plt.scatter([gt[i] for i in idx], [pred[i] for i in idx], alpha=0.6)
        lims = [min(gt + pred) - 0.5, max(gt + pred) + 0.5]
        plt.plot(lims, lims, linestyle="--", color="gray")
        plt.title("SPR_BENCH – Test Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        fname = os.path.join(working_dir, "spr_bench_test_pred_vs_gt.png")
        plt.savefig(fname)
except Exception as e:
    print(f"Error creating scatter plot: {e}")
finally:
    plt.close()

# ---------- print evaluation metrics -----------------------------------------
if val_cplx:
    print(f"Final Validation CplxWA: {val_cplx[-1]:.4f}")
if test_cplx is not None:
    print(f"Test  Complexity-Weighted Accuracy: {test_cplx:.4f}")
