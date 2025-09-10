import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------- Load experiment data -------------------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    data = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

epochs = range(1, len(data.get("losses", {}).get("train", [])) + 1)


# ---------------------- Helper: safe close ---------------------------
def close_fig():
    if plt.get_fignums():
        plt.close()


# --------------------------- Plot 1 ----------------------------------
try:
    plt.figure()
    plt.plot(epochs, data["losses"]["train"], label="Train Loss")
    plt.plot(epochs, data["losses"]["val"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("SPR_BENCH Loss Curve\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve: {e}")
finally:
    close_fig()

# --------------------------- Plot 2 ----------------------------------
try:
    val_compwa = data["metrics"]["val_CompWA"]
    plt.figure()
    plt.plot(epochs, val_compwa, marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("CompWA")
    plt.ylim(0, 1)
    plt.title("SPR_BENCH Validation Complexity-Weighted Accuracy")
    fname = os.path.join(working_dir, "SPR_BENCH_val_CompWA.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating CompWA plot: {e}")
finally:
    close_fig()

# --------------------------- Plot 3 ----------------------------------
try:
    gt = np.array(data["ground_truth"])
    pred = np.array(data["predictions"])
    TP = np.sum((gt == 1) & (pred == 1))
    TN = np.sum((gt == 0) & (pred == 0))
    FP = np.sum((gt == 0) & (pred == 1))
    FN = np.sum((gt == 1) & (pred == 0))
    cm = np.array([[TN, FP], [FN, TP]], dtype=float)
    cm_pct = 100 * cm / max(cm.sum(), 1)
    fig, ax = plt.subplots()
    im = ax.imshow(cm_pct, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm_pct[i, j]:.1f}%", va="center", ha="center", color="black"
            )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    plt.title("SPR_BENCH Confusion Matrix (%)")
    plt.colorbar(im, fraction=0.046)
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    acc = (TP + TN) / max(len(gt), 1)
    print(f"Validation Accuracy: {acc:.4f}")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
finally:
    close_fig()
