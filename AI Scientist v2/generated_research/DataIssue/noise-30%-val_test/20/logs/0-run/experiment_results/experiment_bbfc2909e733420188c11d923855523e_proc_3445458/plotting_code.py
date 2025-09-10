import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}


# Helper to maybe subsample epochs to max 5 points
def epoch_idx(n, max_points=5):
    if n <= max_points:
        return np.arange(n)
    step = max(1, int(np.ceil(n / max_points)))
    return np.arange(0, n, step)


# ----------------------------------------
# 1) Loss curves
try:
    epochs = np.arange(1, len(exp["losses"]["train"]) + 1)
    plt.figure()
    plt.plot(epochs, exp["losses"]["train"], label="Train", color="tab:blue")
    plt.plot(epochs, exp["losses"]["val"], label="Validation", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Train vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ----------------------------------------
# 2) Macro-F1 over epochs
try:
    f1_vals = [m["macro_f1"] for m in exp["metrics"]["val"]]
    ep_sel = epoch_idx(len(f1_vals))
    plt.figure()
    plt.plot(np.arange(1, len(f1_vals) + 1)[ep_sel], np.array(f1_vals)[ep_sel])
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Validation Macro-F1")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_macro_f1_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Macro-F1 curve: {e}")
    plt.close()

# ----------------------------------------
# 3) Complexity-Weighted Accuracy over epochs
try:
    cwa_vals = [m["cwa"] for m in exp["metrics"]["val"]]
    ep_sel = epoch_idx(len(cwa_vals))
    plt.figure()
    plt.plot(np.arange(1, len(cwa_vals) + 1)[ep_sel], np.array(cwa_vals)[ep_sel])
    plt.xlabel("Epoch")
    plt.ylabel("CWA")
    plt.title("SPR_BENCH: Validation Complexity-Weighted Accuracy")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_cwa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CWA curve: {e}")
    plt.close()

# ----------------------------------------
# 4) Final-epoch Macro-F1 vs CWA bar plot
try:
    final_f1 = f1_vals[-1]
    final_cwa = cwa_vals[-1]
    plt.figure()
    plt.bar(["Macro-F1", "CWA"], [final_f1, final_cwa], color=["tab:green", "tab:red"])
    for x, y in zip(["Macro-F1", "CWA"], [final_f1, final_cwa]):
        plt.text(x, y + 0.005, f"{y:.3f}", ha="center", va="bottom")
    plt.ylim(0, 1)
    plt.title("SPR_BENCH: Final-Epoch Metrics")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_metrics_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final metrics bar plot: {e}")
    plt.close()

# ----------------------------------------
# 5) Confusion matrix
try:
    preds = np.array(exp["predictions"])
    gts = np.array(exp["ground_truth"])
    cm = np.zeros((2, 2), dtype=int)
    for p, t in zip(preds, gts):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
    )
    plt.colorbar()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ----------------------------------------
# Print final metrics
try:
    print(f"Final Macro-F1: {final_f1:.4f} | Final CWA: {final_cwa:.4f}")
except Exception as e:
    print(f"Error printing final metrics: {e}")
