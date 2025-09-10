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
    spr_data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = None

if spr_data:
    losses_tr = spr_data["losses"]["train"]
    losses_val = spr_data["losses"]["val"]
    hwa_vals = spr_data["metrics"]["val"]
    preds_last = spr_data["predictions"][-1]
    gts_last = spr_data["ground_truth"][-1]
    epochs = np.arange(1, len(losses_tr) + 1)

    # ---------- plot 1: loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train Loss")
        plt.plot(epochs, losses_val, label="Val Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train Loss, Right: Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- plot 2: validation HWA ----------
    try:
        plt.figure()
        plt.plot(epochs, hwa_vals, marker="o")
        plt.title("SPR_BENCH Validation Harmonic-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        fname = os.path.join(working_dir, "SPR_BENCH_HWA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # ---------- plot 3: prediction vs ground truth distribution ----------
    try:
        plt.figure()
        classes = sorted(list(set(gts_last)))
        gt_counts = [gts_last.count(c) for c in classes]
        pred_counts = [preds_last.count(c) for c in classes]
        x = np.arange(len(classes))
        width = 0.35
        plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_counts, width, label="Predictions")
        plt.title(
            "SPR_BENCH Class Distribution\nLeft: Ground Truth, Right: Predictions"
        )
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.xticks(x, classes)
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_class_distribution.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating distribution plot: {e}")
        plt.close()

    # ---------- console metrics ----------
    final_hwa = hwa_vals[-1] if hwa_vals else None
    best_hwa = np.max(hwa_vals) if hwa_vals else None
    print(f"Final Validation HWA: {final_hwa:.4f}")
    print(f"Best Validation HWA:  {best_hwa:.4f}")
