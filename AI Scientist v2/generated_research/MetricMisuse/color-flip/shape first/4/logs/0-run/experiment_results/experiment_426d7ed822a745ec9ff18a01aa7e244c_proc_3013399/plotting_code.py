import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data is not None:
    losses_tr = data["losses"]["train"]
    losses_val = data["losses"]["val"]
    metrics_val = data["metrics"]["val"]  # list of dicts
    swa = [m["swa"] for m in metrics_val]
    cwa = [m["cwa"] for m in metrics_val]
    hwa = [m["hwa"] for m in metrics_val]
    epochs = list(range(1, len(losses_tr) + 1))
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])
    num_cls = len(np.unique(np.concatenate([preds, gts])))
else:
    losses_tr = losses_val = swa = cwa = hwa = epochs = preds = gts = []
    num_cls = 0

# -------------------- Plot 1: Loss curves --------------------------------
try:
    plt.figure()
    plt.plot(epochs, losses_tr, label="Train Loss")
    plt.plot(epochs, losses_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# -------------------- Plot 2: Metric curves ------------------------------
try:
    plt.figure()
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, hwa, label="HWA")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("SPR_BENCH: Validation Weighted Accuracies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_weighted_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric curve plot: {e}")
    plt.close()

# -------------------- Plot 3: Confusion matrix ---------------------------
try:
    cm = np.zeros((num_cls, num_cls), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SPR_BENCH: Confusion Matrix (Final Epoch)")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------------------- Plot 4: Final metrics bar chart --------------------
try:
    final_vals = [swa[-1], cwa[-1], hwa[-1]] if swa else []
    plt.figure()
    plt.bar(
        ["SWA", "CWA", "HWA"], final_vals, color=["tab:blue", "tab:orange", "tab:green"]
    )
    plt.ylim(0, 1)
    plt.title("SPR_BENCH: Final Epoch Weighted Accuracies")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final metrics bar chart: {e}")
    plt.close()

# -------------------- Console summary ------------------------------------
if swa:
    print(f"Final Epoch - SWA: {swa[-1]:.3f}, CWA: {cwa[-1]:.3f}, HWA: {hwa[-1]:.3f}")
